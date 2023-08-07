import os
import dataclasses

import numpy as np
import torch
from torchaudio.transforms import MelSpectrogram

from musion.util.base import MusionBase, FeatConfig
from musion.util.tools import enframe, deframe
from musion.transcribe.regression import RegressionPostProcessor, events_to_midi

MODULE_PATH = os.path.dirname(__file__)

class Transcribe(MusionBase):
    def __init__(self) -> None:
        super().__init__(
            True,
            MelSpectrogram(**dataclasses.asdict(self._feat_cfg)),
            os.path.join(MODULE_PATH, 'transcribe.onnx'))

    @property
    def _feat_cfg(self) -> FeatConfig:
        return FeatConfig(
            sample_rate=16000,
            n_fft=2048,
            hop_length=160,
            f_min=30,
            f_max=8000,
            n_mels=229,
            norm="slaney",
            mel_scale="slaney",
        )

    def _process(self, samples: np.ndarray) -> dict:
        segs = enframe(samples.squeeze(), self._feat_cfg.sample_rate * 10)

        result = self._batch_process(self._process_segment, segs)

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
