import os
import dataclasses
from typing import Optional

import numpy as np
import torch
from torchaudio.transforms import MelSpectrogram

from musion.util.base import OrtMusionBase, FeatConfig, MusionPCM, TaskDispatcher
from musion.util.tools import normalize, median_filter

MODULE_PATH = os.path.dirname(__file__)

class Struct(TaskDispatcher):
    def __init__(self) -> None:
        super().__init__(_Struct)

class _Struct(OrtMusionBase):
    def __init__(self, device: str = None) -> None:
        super().__init__(
            os.path.join(MODULE_PATH, 'struct.onnx'),
            device)
        mel_spec_cfg = dataclasses.asdict(self._feat_cfg)
        mel_spec_cfg.pop('mono')
        self._feat = MelSpectrogram(**mel_spec_cfg).to(self.device)

    @property
    def _feat_cfg(self) -> FeatConfig:
        return FeatConfig(
            mono=True,
            sample_rate=22050,
            n_fft=2048,
            hop_length=512,
            f_min=20,
            f_max=5000,
            n_mels=128,
            power=1,
            norm="slaney"
        )

    def _process(self, audio_path: Optional[str] = None, pcm: Optional[MusionPCM] = None) -> dict:
        pcm = self._load_pcm(audio_path, pcm)
        feat = self._feat(torch.from_numpy(pcm.samples).to(self.device)).cpu().numpy()
        num_frames = feat.shape[-1]
        remain = num_frames % 9
        remain_np = np.zeros([128, 9 - remain, 1])
        feature_crop = np.concatenate((feat[0][:, :, None], remain_np), axis=1)
        feature_crop = np.expand_dims(feature_crop, axis=0)

        res = self._predict(feature_crop)[0].squeeze()

        res = normalize(res)
        res = median_filter(res, 9)

        res = self.__get_chorus_segments(res)

        return {'struct': res}

    def __get_chorus_segments(self, chorus_prob_per_sec):
        chorus_raw = chorus_prob_per_sec > 0.5
        chorus_segmens = []
        in_seg = False
        for i in range(len(chorus_raw)):
            if chorus_raw[i] and not in_seg:
                chorus_segmens.append([i])
                in_seg = True
            elif not chorus_raw[i] and in_seg:
                chorus_segmens[-1].append(i)
                in_seg = False

        return chorus_segmens

    @property
    def result_keys(self):
        return ['struct']
