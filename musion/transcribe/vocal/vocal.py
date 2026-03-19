from typing import Optional
import os

import numpy as np
import torch
from torchaudio.transforms import Resample
import pretty_midi

import musion
from musion.transcribe.base import *
from musion.utils.base import FeatConfig, MusionPCM
from musion.transcribe.midi_utils import convert_to_type0

MODULE_PATH = os.path.dirname(__file__)

class _VocalTranscribe(TranscribeBase):
    def __init__(self, device: str = None) -> None:
        super().__init__(os.path.join(MODULE_PATH, 'transcribe_vocal.onnx'), device)
        self.separate = musion.separate._Separate(self.device)
        self.resampler = Resample(self.separate._feat_cfg.sample_rate, self._feat_cfg.sample_rate).to(self.device)

    @property
    def _feat_cfg(self) -> FeatConfig:
        return FeatConfig(
            mono=True,
            sample_rate=16000,
            n_fft=None,
            hop_length=None
        )

    @property
    def accept_input_stem(self) -> str:
        return 'vocals'

    def _process_wo_beat_align(self, audio_path: Optional[str] = None, pcm: Optional[MusionPCM] = None) -> pretty_midi.PrettyMIDI:
        sig = self._load_pcm(audio_path, pcm).samples.squeeze()

        sig_list = []
        seg_size = 5 * self._feat_cfg.sample_rate
        for i in range(0, len(sig), seg_size):
            sig_list.append(sig[i:i+seg_size])
        if len(sig_list[-1]) < seg_size:
            sig_list[-1] = np.pad(sig_list[-1], (0, seg_size - len(sig_list[-1])))

        song_pred = []
        pitch_octave_num = 4

        for sig in sig_list:
            sig = sig[None, :] # [batch, wav_len]
            logits = self._predict(sig)
            logits = np.asarray(logits[0])

            onset_logits = logits[:, :, 0]
            offset_logits = logits[:, :, 1]
            pitch_out = logits[:, :, 2:]
            pitch_octave_logits = pitch_out[:, :, 0:pitch_octave_num+1]
            pitch_class_logits = pitch_out[:, :, pitch_octave_num+1:]

            batch_size, frame = onset_logits.shape[:2]
            onset_logits = torch.from_numpy(onset_logits)
            offset_logits = torch.from_numpy(offset_logits)
            onset_probs, offset_probs = torch.sigmoid(onset_logits[0]), torch.sigmoid(offset_logits[0])
            onset_probs = onset_probs.cpu().numpy()
            offset_probs = offset_probs.cpu().numpy()
            pitch_octave_logits, pitch_class_logits = pitch_octave_logits[0], pitch_class_logits[0]
            for f in range(frame):
                frame_info = (
                    onset_probs[f], offset_probs[f], np.argmax(pitch_octave_logits[f]).item(),
                    np.argmax(pitch_class_logits[f]).item()
                )
                song_pred.append(frame_info)
        est_result = self.__frame2note(song_pred, onset_thres=0.4, 
                        offset_thres=0.5)

        midi = self.note_seq_to_pretty_midi(est_result)
        return midi

    @staticmethod
    def __frame2note(frame_info, onset_thres, offset_thres, frame_size=1/49.8):
        """
        This function transforms the frame-level predictions into the note-level predictions.
        Parse frame info [(onset_probs, offset_probs, pitch_class)...] into desired label format.
        Adapted from https://github.com/york135/singing_transcription_ICASSP2021/blob/master/AST/predictor.py
        """

        result = []
        current_onset = None
        pitch_counter = []

        onset_seq = np.array([frame_info[i][0] for i in range(len(frame_info))])

        local_max_size = 3
        current_time = 0.0

        onset_seq_length = len(onset_seq)

        for i in range(len(frame_info)):
            current_time = frame_size*i
            info = frame_info[i]

            backward_frames = i - local_max_size
            if backward_frames < 0:
                backward_frames = 0

            forward_frames = i + local_max_size + 1
            if forward_frames > onset_seq_length - 1:
                forward_frames = onset_seq_length - 1

            # local max and more than threshold
            if info[0] >= onset_thres and onset_seq[i] == np.amax(onset_seq[backward_frames : forward_frames]):
                if current_onset is None:
                    current_onset = current_time
                else:
                    if len(pitch_counter) > 0:
                        result.append([current_onset, current_time, max(set(pitch_counter), key=pitch_counter.count) + 36])
                    current_onset = current_time
                    pitch_counter = []
            elif info[1] >= offset_thres:  # If is offset
                if current_onset is not None:
                    if len(pitch_counter) > 0:
                        result.append([current_onset, current_time, max(set(pitch_counter), key=pitch_counter.count) + 36])
                    current_onset = None

                    pitch_counter = []
            # If current_onset exist, add count for the pitch
            if current_onset is not None:
                final_pitch = int(info[2]* 12 + info[3])
                if info[2] != 4 and info[3] != 12:
                    pitch_counter.append(final_pitch)

        if current_onset is not None:
            if len(pitch_counter) > 0:
                result.append([current_onset, 
                               current_time, 
                               int(max(set(pitch_counter), key=pitch_counter.count)) + 36])
            current_onset = None

        return result

    def note_seq_to_pretty_midi(self, note_seq):
        """
        note_seq: [[onset_time, offset_time, pitch], ...]
        """
        midi = pretty_midi.PrettyMIDI()
        instrument = pretty_midi.Instrument(program=65) # use 65(alto sax) to represent vocal
        for note in note_seq:
            note = pretty_midi.Note(90, note[2], note[0], note[1])
            instrument.notes.append(note)
        midi.instruments.append(instrument)
        return midi
