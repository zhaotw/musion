import os
import dataclasses

import numpy as np
import torch
from torchaudio.transforms import MelSpectrogram

from musion.util.base import MusionBase, FeatConfig
from musion.util.tools import normalize, median_filter

MODULE_PATH = os.path.dirname(__file__)

class Struct(MusionBase):
    def __init__(self) -> None:
        super().__init__(
            True,
            MelSpectrogram(**dataclasses.asdict(self._feat_cfg)),
            os.path.join(MODULE_PATH, 'struct.onnx'))

    @property
    def _feat_cfg(self) -> FeatConfig:
        return FeatConfig(
            sample_rate=22050,
            n_fft=2048,
            hop_length=512,
            f_min=20,
            f_max=5000,
            n_mels=128,
            power=1,
            norm="slaney"
        )

    def _process(self, samples: np.ndarray) -> dict:
        feat = self._feat(torch.from_numpy(samples).to(self.device)).cpu().numpy()
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
