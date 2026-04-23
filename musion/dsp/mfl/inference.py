import os
from typing import Any, Optional

import torch
from torchaudio.transforms import Spectrogram

from musion.utils.base import MusionBase, FeatConfig, MusionPCM, TaskDispatcher
from musion.separate.inference import _Separate


MODULE_PATH = os.path.dirname(__file__)


class MidFocusLevel(TaskDispatcher):
    def __init__(self) -> None:
        super().__init__(_MidFocusLevel)


class _MidFocusLevel(MusionBase):
    def __init__(self, device: str = None, **kwargs: Any) -> None:
        super().__init__(device)

        self.separate = _Separate(self.device)
        self.spec_proc = Spectrogram(
            n_fft=self._feat_cfg.n_fft,
            power=self._feat_cfg.power,
        ).to(self.device)

        self.freq_slice = slice(self.freq2fft(200), self.freq2fft(7000))

        self.kernel_size = self._feat_cfg.sample_rate // 1000
        self.conv_kernel = (
            torch.ones((1, 1, self.kernel_size)).to(self.device) / self.kernel_size
        )

    @property
    def _feat_cfg(self) -> FeatConfig:
        return FeatConfig(
            mono=False,
            sample_rate=44100,
            n_fft=2048,
            power=2,
        )

    def freq2fft(self, freq: float) -> int:
        return int(freq * self._feat_cfg.n_fft / self._feat_cfg.sample_rate)

    def filter_spec(self, samples: torch.Tensor) -> torch.Tensor:
        spec = self.spec_proc(samples)
        spec = spec[self.freq_slice, :]

        return spec

    def _process(
        self, audio_path: Optional[str] = None, pcm: Optional[MusionPCM] = None
    ) -> dict:
        separated = self.separate(audio_path=audio_path)

        samples = separated["other.wav"].to(self.device)
        side = (samples[1] - samples[0]) * 0.5

        spec_mid = (
            self.filter_spec(samples[0]) + self.filter_spec(samples[1])
        ) / 2
        spec_side = self.filter_spec(side)

        mid_energy_ratios = []
        for i in range(spec_mid.shape[1]):
            energy_side_seg = torch.sum(spec_side[:, i])
            if energy_side_seg < 1:
                mid_energy_ratios.append(0)
                continue

            energy_mid_seg = torch.sum(spec_mid[:, i])
            mid_energy_ratio = 1 - energy_side_seg / energy_mid_seg
            mid_energy_ratios.append(mid_energy_ratio)

        mid_energy_ratios = torch.as_tensor(mid_energy_ratios).to(self.device)
        if not mid_energy_ratios.any():
            return {"mid_focus_level": 1}

        mid_energy_ratios = mid_energy_ratios.unsqueeze(0).unsqueeze(0)
        mid_energy_ratios = torch.conv1d(
            mid_energy_ratios,
            self.conv_kernel,
            stride=self.kernel_size,
        )

        mid_energy_ratios = mid_energy_ratios[0][0]

        mfl = (
            torch.sum(mid_energy_ratios > 0.7).item() / len(mid_energy_ratios)
        )

        return {"mid_focus_level": mfl}

    @property
    @staticmethod
    def result_keys(self):
        return ["mid_focus_level"]
