import os
import dataclasses
from typing import Optional

import numpy as np
import torch
from torchaudio.transforms import MelSpectrogram, Resample

import musion
from musion.utils.base import FeatConfig, TaskDispatcher, MusionPCM
from musion.utils.ort_musion_base import OrtMusionBase
from musion.utils.tools import enframe, deframe
from musion.transcribe.regression import RegressionPostProcessor, events_to_midi

MODULE_PATH = os.path.dirname(__file__)

class Transcribe(TaskDispatcher): 
    def __init__(self, target_instrument: str):
        if target_instrument == 'piano':
            task_class = _PianoTranscribe
        elif target_instrument == 'vocal':
            task_class = _VocalTranscribe
        else:
            raise ValueError(f"Unsupported instrument: {target_instrument}")
        super().__init__(task_class)

class _PianoTranscribe(OrtMusionBase):
    def __init__(self, device: str = None) -> None:
        super().__init__(
            os.path.join(MODULE_PATH, 'transcribe_piano.onnx'),
            device)
        mel_spec_cfg = dataclasses.asdict(self._feat_cfg)
        mel_spec_cfg.pop('mono')
        self._feat = MelSpectrogram(**mel_spec_cfg).to(self.device)

    @property
    def _feat_cfg(self) -> FeatConfig:
        return FeatConfig(
            mono=True,
            sample_rate=16000,
            n_fft=2048,
            hop_length=160,
            f_min=30,
            f_max=8000,
            n_mels=229,
            norm="slaney",
            mel_scale="slaney",
        )

    def _process(self, audio_path: Optional[str] = None, pcm: Optional[MusionPCM] = None) -> dict:
        pcm = self._load_pcm(audio_path, pcm)
        segs = enframe(pcm.samples.squeeze(), self._feat_cfg.sample_rate * 5, self._feat_cfg.sample_rate * 10)

        result = self._batch_process(self._process_segment, segs, 2)

        output_dict_names = [
            'reg_onset_output', 'reg_offset_output', 'frame_output', 'velocity_output',
            'reg_pedal_onset_output', 'reg_pedal_offset_output', 'pedal_frame_output'
        ]
        output_dict = {i: [] for i in output_dict_names}
        for seg in result:
            for key, value in zip(output_dict_names, seg):
                output_dict[key].append(value)

        for key in output_dict.keys():
            output_dict[key] = deframe(np.asarray(output_dict[key]))

        (est_note_events, est_pedal_events) = RegressionPostProcessor().output_dict_to_midi_events(output_dict)

        midi = events_to_midi(est_note_events, est_pedal_events)

        return {'mid': midi}

    def _process_segment(self, segment):
        feat = self._feat(torch.from_numpy(segment).to(self.device))
        feat = 10.0 * torch.log10(torch.clamp(feat, min=1e-10, max=np.inf)) # power to dB
        return [segment for segment in self._predict(feat[:, :, :, None].cpu().numpy())[0]]

    @property
    def result_keys(self):
        return ['mid']
    
    def _save(self, key, save_path, res):
        if 'mid' in key:
            res[key].save(save_path)

class _VocalTranscribe(OrtMusionBase):
    def __init__(self, device: str = None) -> None:
        super().__init__(
            os.path.join(MODULE_PATH, 'transcribe_vocal.onnx'),
            device)
        self.separate = musion.separate._Separate(device)
        self.resampler = Resample(self.separate._feat_cfg.sample_rate, self._feat_cfg.sample_rate).to(self.device)

    @property
    def _feat_cfg(self) -> FeatConfig:
        return FeatConfig(
            mono=True,
            sample_rate=16000,
            n_fft=None,
            hop_length=None
        )

    def _process(self, audio_path: Optional[str] = None, pcm: Optional[MusionPCM] = None) -> dict:
        vocals = self.separate(audio_path=audio_path, pcm=pcm)['vocals.wav'].to(self.device)
        sig = vocals.mean(dim=0, keepdim=False)
        sig = self.resampler(sig)

        sig_list = []
        seg_size = 5 * self._feat_cfg.sample_rate
        for i in range(0, len(sig), seg_size):
            sig_list.append(sig[i:i+seg_size].cpu().numpy())
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

        return {"vocals": est_result}

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
                               max(set(pitch_counter), key=pitch_counter.count) + 36])
            current_onset = None

        return result

    @property
    @staticmethod
    def result_keys(self):
        return ["vocals"]
