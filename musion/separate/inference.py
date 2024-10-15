import os
import math
from typing import Optional

import torch
from torchaudio.transforms import Spectrogram
from torch.nn import functional as F

from musion.util.base import MusionBase, FeatConfig, MusionPCM, TaskDispatcher
from .utils import *

MODULE_PATH = os.path.dirname(__file__)

class Separate(TaskDispatcher):
    def __init__(self, num_workers: int = 1) -> None:
        super().__init__(_Separate(), num_workers)

class _Separate(MusionBase):
    SOURCES = ["drums", "bass", "other", "vocals"]

    def __init__(self, device: str = None) -> None:
        super().__init__(
            False,
            os.path.join(MODULE_PATH, 'separate.onnx'),
            device)
        self._feat = Spectrogram(
                n_fft=self._feat_cfg.n_fft,
                hop_length=self._feat_cfg.hop_length,
                win_length=self._feat_cfg.n_fft,
                normalized=True,
                center=True
            ).to(self.device)
        self.training_length = 343980 # 39 / 5 * 44100

    @property
    def _feat_cfg(self) -> FeatConfig:
        return FeatConfig(
            sample_rate=44100,
            n_fft=4096,
            hop_length=1024,
        )

    def _process(self, audio_path: Optional[str] = None, pcm: Optional[MusionPCM] = None) -> dict:
        samples = self._load_pcm(audio_path, pcm).samples
        channels, length = samples.shape

        # standardize
        ref = samples.mean(0)
        samples -= ref.mean()
        samples /= ref.std()

        out = torch.zeros((1, 4, channels, length), device=self.device)
        sum_weight = torch.zeros(length, device=self.device)

        segment_length: int = self.training_length
        stride = int(0.75 * segment_length)
        offsets = range(0, length, stride)

        # We start from a triangle shaped weight, with maximal weight in the middle
        # of the segment. Then we normalize and take to the power `transition_power`.
        # Large values of transition power will lead to sharper transitions.
        weight = torch.cat([torch.arange(1, segment_length // 2 + 1, device=self.device),
                        torch.arange(segment_length - segment_length // 2, 0, -1, device=self.device)])

        weight = weight / weight.max()
        segments = []
        offset_and_len = []
        for offset in offsets:
            chunk = TensorChunk(torch.from_numpy(samples), offset, segment_length)
            segments.append(chunk.padded(self.training_length))
            offset_and_len.append((offset, chunk.length))

        res = self._batch_process(self._process_segment, segments)

        for chunk_out, (offset, len) in zip(res, offset_and_len):
            chunk_out = center_trim(chunk_out.to(self.device), len)
            chunk_length = chunk_out.shape[-1]
            out[..., offset:offset + segment_length] += (
                weight[:chunk_length] * chunk_out)
            sum_weight[offset:offset + segment_length] += weight[:chunk_length]

        out /= sum_weight

        out *= ref.std()
        out += ref.mean()
        out = out[0]
        for id, src in enumerate(out):
            out[id] /= max(1.01 * out[id].abs().max(), 1)

        return dict(zip(self.result_keys, out.cpu()))

    def _spec(self, x):
        hl = self._feat_cfg.hop_length

        le = int(math.ceil(x.shape[-1] / hl))
        pad = hl // 2 * 3
        x = pad1d(x, (pad, pad + le * hl - x.shape[-1]), mode="reflect")

        z = spectro(x, self._feat_cfg.n_fft, hl)[..., :-1, :]

        z = z[..., 2: 2 + le]
        return z

    def _ispec(self, z, length=None, scale=0):
        hl = self._feat_cfg.hop_length // (4**scale)
        z = F.pad(z, (0, 0, 0, 1))
        z = F.pad(z, (2, 2))
        pad = hl // 2 * 3
        le = hl * int(math.ceil(length / hl)) + 2 * pad
        x = ispectro(z, hl, length=le)
        x = x[..., pad: pad + length]
        return x


    def _process_segment(self, mix_np):
        mix = torch.as_tensor(mix_np).to(self.device)
        z = self._spec(mix)

        mag = magnitude(z).cpu()
        B, C, Fq, T = mag.shape

        x, xt = self._predict([mix_np, mag.numpy()])
        x, xt = torch.from_numpy(x).to(self.device), torch.from_numpy(xt).to(self.device)

        x_is_mps = x.device.type == "mps"
        if x_is_mps:
            x = x.cpu()

        zout = mask(x)

        x = self._ispec(zout, self.training_length)

        # back to mps device
        if x_is_mps:
            x = x.to("mps")

        meant = mix.mean(dim=(1, 2), keepdim=True)
        stdt = mix.std(dim=(1, 2), keepdim=True)
        xt = xt.view(B, len(self.SOURCES), -1, self.training_length)
        xt = xt * stdt[:, None] + meant[:, None]
        x = xt + x

        return x

    @property
    def result_keys(self):
        return [s + '.wav' for s in self.SOURCES]
