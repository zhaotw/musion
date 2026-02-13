import os
import dataclasses
from typing import Optional, List, Tuple
import math
from collections import defaultdict

import torch
import pretty_midi

from musion.utils.base import FeatConfig, MusionPCM
from musion.utils.ort_musion_base import OrtMusionBase
from musion.transcribe.base import TranscribeBase
from musion.transcribe.midi_utils import MIDI_RESOLUTION
from musion.transcribe.piano.feat import *

MODULE_PATH = os.path.dirname(__file__)

@dataclasses.dataclass
class Note:
    start: float
    end: float
    pitch: int
    velocity: int
    hasOnset: bool
    hasOffset: bool

def listToIdx(l):
    batchIndices = [ idx for idx, curList in enumerate(l) for _ in curList]  

    return batchIndices

@torch.jit.script
def viterbiBackward(score, forcedStartPos: List[int]):
    # score: [nEndPos,  nBeginPos, nTargetPitch]
    T = score.shape[0]
    nTargetPitch = score.shape[2]

    q = torch.zeros(T, nTargetPitch, device = score.device)

    # for back tracking
    ptr = []

    scoreT = score.transpose(0,1).contiguous()

    q[T-1] = score[T-1, T-1, :] * (score[T-1, T-1, :] > 0)

    for i in range(1,T):
        subScore = scoreT[T-i-1, T-i:, :]

        candidate_scores = torch.cat(
            [
            q[None, T-i, :],   # skip
            q[T-i:, :]+ subScore  # an interval 
            ],
            dim = 0
        )
        
        curV, selection = candidate_scores.max(dim = 0)

        ptr.append(selection-1)
        singletonMask = score[T-i-1, T-i-1,:]>0

        q[T-i-1] = curV+ score[T-i-1,T-i-1,:] * singletonMask

    ptr = torch.stack(ptr, dim = 0).cpu()

    scoreDiagInclusion = (torch.diagonal(score, dim1= 0, dim2=1)>0).cpu()

    # perform backtracking 
    result: List[List[Tuple[int, int]]]  =  []

    for idx in range(nTargetPitch):
        j = forcedStartPos[idx]

        curResult : List[Tuple[int, int]]  = []
        curDiag = scoreDiagInclusion[idx]
        while j < T-1:
            curSelecton= int(ptr[T-j-2][idx])
            if bool(curDiag[j]):
                curResult.append((j,j))

            if curSelecton<0:
                j += 1
            else:
                i = curSelecton+j+1
                curResult.append((j,i))
                j = i
    
        if score[T-1,T-1, idx]>0:
            curResult.append((T-1,T-1))

        result.append(curResult)

    return result

class _PianoHeads(nn.Module):
    def __init__(self):
        super().__init__()
        self.velocityPredictor = nn.Sequential(
                                nn.Linear(768, 512),
                                nn.GELU(),
                                # nn.Dropout(0),
                                nn.Linear(512, 128)
                                )
        self.refinedOFPredictor = nn.Sequential(
                                nn.Linear(768, 512),
                                nn.GELU(),
                                # nn.Dropout(0),
                                nn.Linear(512, 4)
                                )

        checkpoint = torch.load(os.path.join(MODULE_PATH, 'transcribe_piano_heads.pt'))
        self.load_state_dict(checkpoint, strict=True)

        self.eval()
    
    def forward(self, x):
        logitsVelocity = self.velocityPredictor(x)
        pVelocity = F.softmax(logitsVelocity, dim = -1)
        velocity = torch.argmax(pVelocity, dim = -1)

        ofValue, ofPresence = self.refinedOFPredictor(x).chunk(2, dim = -1)
        ofDist = torch.distributions.ContinuousBernoulli(logits=ofValue)

        ofValue = (ofDist.mean-0.5)/0.99
        ofValue = torch.clamp(ofValue, -0.5, 0.5)
        ofPresence = ofPresence > 0

        return velocity, ofValue, ofPresence

class _PianoTranscribe(TranscribeBase, OrtMusionBase):
    def __init__(self, device: str = None) -> None:
        TranscribeBase.__init__(self, device)
        OrtMusionBase.__init__(self,
            os.path.join(MODULE_PATH, 'transcribe_piano.onnx'),
            device)

        mel_spec_cfg = dataclasses.asdict(self._feat_cfg)
        mel_spec_cfg.pop('mono')
        mel_spec_cfg.pop('hop_length')

        self.framewiseFeatureExtractor = MelSpectrum(**mel_spec_cfg).to(self.device)

        self.targetMIDIPitch = [-64, -67] + list(range(21, 108+1))

        self._PianoHeads = _PianoHeads().to(self.device)

        stepInSecond = 8
        segmentSizeInSecond = 16

        self.padTimeBegin = (segmentSizeInSecond-stepInSecond)
        self.mix_pad = self.padTimeBegin * self._feat_cfg.sample_rate

        self.startFrameIdx = math.floor(self.mix_pad / self._feat_cfg.hop_length)        

        self.stepSize = math.ceil(stepInSecond*self._feat_cfg.sample_rate/self._feat_cfg.hop_length)*self._feat_cfg.hop_length
        self.segmentSize = math.ceil(segmentSizeInSecond*self._feat_cfg.sample_rate)
        self.lastFrameIdx = round(self.segmentSize/self._feat_cfg.hop_length) # 689

    @property
    def _feat_cfg(self) -> FeatConfig:
        return FeatConfig(
            mono=False,
            sample_rate=44100,
            n_fft=4096,
            hop_length=1024,
            f_min=30,
            f_max=8000,
            n_mels=229,
        )

    def _process(self, audio_path: Optional[str] = None, pcm: Optional[MusionPCM] = None) -> dict:
        x = self._load_pcm(audio_path, pcm).samples
        x = torch.as_tensor(x, device=self.device)
        x = F.pad(x, (self.mix_pad, self.mix_pad))

        eventsByType = defaultdict(list)
        startPos = [self.startFrameIdx] * len(self.targetMIDIPitch)
        nSample = x.shape[-1]
        for i in range(0, nSample, self.stepSize):
            j = min(i + self.segmentSize, nSample)

            beginTime = i / self._feat_cfg.sample_rate - self.padTimeBegin

            curSlice = x[:, i:j]

            if curSlice.shape[-1]< self.segmentSize:
                # pad to the segmentSize
                curSlice = F.pad(curSlice, (0, self.segmentSize-curSlice.shape[-1]))

            curFrames = makeFrame(curSlice, self._feat_cfg.hop_length, self._feat_cfg.n_fft)

            notes, startPos = self.transcribe_segment(curFrames.unsqueeze(0), forcedStartPos=startPos)
            notes = notes[0]

            # # shift all notes by beginTime
            for e in notes:
                e.start += beginTime
                e.end  += beginTime

            for e in notes:
                if len(eventsByType[e.pitch])>0:
                    last_e = eventsByType[e.pitch][-1]

                    # test if e overlap with the last event 
                    if e.start < last_e.end:
                        if e.hasOnset: #and e.hasOffset:
                            eventsByType[e.pitch][-1] = e
                        else:
                            # merge two events
                            eventsByType[e.pitch][-1].hasOffset = e.hasOffset
                            eventsByType[e.pitch][-1].end = max(e.end, last_e.end)

                        continue

                if e.hasOnset:
                    eventsByType[e.pitch].append(e)

        # handling incomplete events in the last segment
        for eventType in eventsByType:
            if len(eventsByType[eventType])>0:
                eventsByType[eventType][-1].hasOffset = True

        # flatten all events
        eventsAll = sum(eventsByType.values(), [])

        # post filtering
        eventsAll = [n for n in eventsAll if n.hasOffset]

        eventsAll = self.resolveOverlapping(eventsAll)

        midi = self.events_to_midi(eventsAll, MIDI_RESOLUTION)
        midi = self._align_midi_with_beats(midi, audio_path)
        return {'mid': midi}

    def events_to_midi(self, notes, resolution): 
        outputMidi = pretty_midi.PrettyMIDI(resolution=resolution)

        piano_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
        piano = pretty_midi.Instrument(program = piano_program)

        for note in notes:
            if note.pitch > 0:
                note = pretty_midi.Note(start = note.start,
                                        end = note.end,
                                        pitch = note.pitch,
                                        velocity = note.velocity)
                piano.notes.append(note)
            else:
                cc_on = pretty_midi.ControlChange(-note.pitch, note.velocity, note.start)
                cc_off = pretty_midi.ControlChange(-note.pitch, 0, note.end)

                piano.control_changes.append(cc_on)
                piano.control_changes.append(cc_off)

        outputMidi.instruments.append(piano)
        return outputMidi

    def resolveOverlapping(self, note_events):
        note_events.sort(key = lambda x: (x.start, x.end,x.pitch))

        ex_note_events = []
        idx = 0
        buffer_dict = {}
        
        for note_event in note_events:
            midi_note = note_event.pitch

            if midi_note in buffer_dict.keys():
                _idx = buffer_dict[midi_note]
                if ex_note_events[_idx].end > note_event.start:
                    raise
                    ex_note_events[_idx].end = note_event.start
            
            buffer_dict[midi_note] = idx
            idx += 1

            ex_note_events.append(note_event)

        ex_note_events.sort(key = lambda x: (x.start, x.end,x.pitch))

        return ex_note_events

    def fetchIntervalFeaturesBatch(self, ctxBatch, intervalsBatch):
        # ctx: [N, SYM, T, D]
        ctx_a_all = []
        ctx_b_all = []
        device = ctxBatch.device
        T = ctxBatch.shape[-2]

        for idx, curIntervals in enumerate(intervalsBatch):
            nIntervals =len(sum(curIntervals, []))
            if nIntervals>0:
                symIdx = torch.tensor(listToIdx(curIntervals), dtype=torch.long, device = device)
                indices = torch.tensor(sum(curIntervals, []), dtype =torch.long, device = device)

                ctx_a = ctxBatch[idx].flatten(0,1).index_select(dim = 0, index = indices[:, 0]+ symIdx*T)
                ctx_b = ctxBatch[idx].flatten(0,1).index_select(dim = 0, index = indices[:, 1]+ symIdx*T)

                ctx_a_all.append(ctx_a)
                ctx_b_all.append(ctx_b)

        ctx_a_all = torch.cat(ctx_a_all, dim = 0)
        ctx_b_all = torch.cat(ctx_b_all, dim = 0)

        attributeInput = torch.cat([ctx_a_all,
                                   ctx_b_all,
                                   ctx_a_all*ctx_b_all,
                                   ],dim = -1)

        return attributeInput

    def transcribe_segment(self,framesBatch, forcedStartPos):
        nBatch = framesBatch.shape[0]
        intervalsBatch, velocity, ofValue, ofPresence = self.processFramesBatch(framesBatch, forcedStartPos)
        if intervalsBatch is None:
            return [[] for _ in range(nBatch)], [0 for _ in range(len(self.targetMIDIPitch))]

        velocity = velocity.cpu().detach().tolist()
        ofValue = ofValue.cpu().detach().tolist()
        ofPresence = ofPresence.cpu().detach().tolist()
        nCount = 0 

        notes = [[] for _ in range(nBatch)]

        frameDur = self._feat_cfg.hop_length/self._feat_cfg.sample_rate

        # the last offset
        lastP = []

        for idx in range(nBatch):
            curIntervals = intervalsBatch[idx]

            for j, eventType in enumerate(self.targetMIDIPitch):
                lastEnd = 0
                curLastP = 0

                for aInterval in curIntervals[j]: 
                    curVelocity = velocity[nCount]
                    curOffset = ofValue[nCount]
                    start = (aInterval[0] + curOffset[0]) * frameDur
                    end = (aInterval[1] + curOffset[1]) * frameDur

                    # ofPresence prediction is only used to distinguish the corner case that either onset or offset happens exactly on the first/last frame.
                    hasOnset = (aInterval[0]>0) or ofPresence[nCount][0]
                    hasOffset = (aInterval[1]<self.lastFrameIdx) or ofPresence[nCount][1]

                    start = max(start, lastEnd)
                    end = max(end, start+1e-8)
                    lastEnd = end
                    curNote = Note(
                         start = start,
                         end = end,
                         pitch = eventType,
                         velocity = curVelocity,
                         hasOnset = hasOnset,
                         hasOffset = hasOffset)

                    notes[idx].append(curNote)

                    if hasOffset:
                        curLastP = aInterval[1]

                    nCount+= 1

                lastP.append(curLastP)

            notes[idx].sort(key = lambda x: (x.start, x.end, x.pitch))

        new_start_pos = []
        for k in lastP:
            new_start_pos.append(max(k-int(self.stepSize/self._feat_cfg.hop_length), 0))

        return notes, new_start_pos

    def processFramesBatch(self, framesBatch, forcedStartPos):
        # gain normalization
        framesBatchMean = torch.mean(framesBatch, dim=[1,2,3], keepdim=True)
        framesBatchStd = torch.std(framesBatch, dim=[1,2,3], keepdim=True)
        framesBatch = (framesBatch - framesBatchMean)/(framesBatchStd+ 1e-8) # [B, AudioCh, T, winsize]

        featuresBatch = self.framewiseFeatureExtractor(framesBatch).contiguous()
        # now with shape [nBatch, nAudioChannel, nStep, NFreq, nChannel]

        featuresBatch = featuresBatch.view(-1, *featuresBatch.shape[-3:])

        ctx_batch, S_batch, _ = self._predict_torch(featuresBatch)
        S_batch = S_batch.flatten(-2,-1)            # discard batch dim

        # with shape [*, nBatch*nSym]
        with torch.inference_mode():
            path = viterbiBackward(S_batch, forcedStartPos)

        nSymbols = len(self.targetMIDIPitch)

        # then predict attributes associated with frames obtain segment features
        nBatch = framesBatch.shape[0]
        nIntervalsAll = sum([len(_) for _ in path])
        if nIntervalsAll == 0:
            # nothing detected, return empty
            return None, None, None, None

        intervalsBatch = []
        for idx in range(nBatch):
            curIntervals =  path[idx*nSymbols: (idx+1)*nSymbols]
            intervalsBatch.append(curIntervals)

        attributeInput = self.fetchIntervalFeaturesBatch(ctx_batch, intervalsBatch)

        return intervalsBatch, *self._PianoHeads(attributeInput)
