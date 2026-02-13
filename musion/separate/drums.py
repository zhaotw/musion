from typing import Optional
import os

import torch
import numpy as np

from musion.utils.ort_musion_base import OrtMusionBase
from musion.utils.base import MusionPCM, FeatConfig


MODULE_PATH = os.path.dirname(__file__)

class STFT:
    def __init__(self, device: torch.device):
        self.n_fft = 2048
        self.hop_length = 512
        self.device = device
        self.window = torch.hann_window(window_length=self.n_fft, periodic=True, device=device)
        self.dim_f = 1024

    def __call__(self, x):
        batch_dims = x.shape[:-2]
        c, t = x.shape[-2:]
        x = x.reshape([-1, t])
        x = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            center=True,
            return_complex=True
        )
        x = torch.view_as_real(x)
        x = x.permute([0, 3, 1, 2])
        x = x.reshape([*batch_dims, c, 2, -1, x.shape[-1]]).reshape([*batch_dims, c * 2, -1, x.shape[-1]])
        return x[..., :self.dim_f, :]

    def inverse(self, x):
        batch_dims = x.shape[:-3]
        c, f, t = x.shape[-3:]
        n = self.n_fft // 2 + 1
        f_pad = torch.zeros([*batch_dims, c, n - f, t], device=self.device, dtype=x.dtype)
        x = torch.cat([x, f_pad], -2)
        x = x.reshape([*batch_dims, c // 2, 2, n, t]).reshape([-1, 2, n, t])
        x = x.permute([0, 2, 3, 1])
        x = x[..., 0] + x[..., 1] * 1.j
        x = torch.istft(x, n_fft=self.n_fft, hop_length=self.hop_length, window=self.window, center=True)
        x = x.reshape([*batch_dims, 2, -1])
        return x

class _DrumsSeparate(OrtMusionBase):
    def __init__(self, device: str = None) -> None:
        OrtMusionBase.__init__(self,
            os.path.join(MODULE_PATH, 'separate_drums.onnx'),
            device,
            trt_fp16_enable=True)

        self.stft = STFT(self.device)

    @property
    def _feat_cfg(self) -> FeatConfig:
        return FeatConfig(
            mono=True,
            sample_rate=44100,
        )

    def _getWindowingArray(self, window_size, fade_size):
        fadein = torch.linspace(0, 1, fade_size, device=self.device)
        fadeout = torch.linspace(1, 0, fade_size, device=self.device)
        window = torch.ones(window_size, device=self.device)
        window[-fade_size:] *= fadeout
        window[:fade_size] *= fadein
        return window

    def demix_track(self, mix: np.ndarray):
        chunk_size = 130560
        N = 4
        fade_size = chunk_size // 10
        step = int(chunk_size // N)
        border = chunk_size - step
        batch_size = 1

        full_length = mix.shape[-1]

        # Do pad from the beginning and end to account floating window results better
        if full_length > 2 * border:
            mix = np.pad(mix, ((0, 0), (border, border)), mode='reflect')

        # windowingArray crossfades at segment boundaries to mitigate clicking artifacts
        windowingArray = self._getWindowingArray(chunk_size, fade_size)

        req_shape = (6,) + tuple(mix.shape)

        result = torch.zeros(req_shape, dtype=torch.float32, device=self.device)
        counter = torch.zeros(req_shape, dtype=torch.float32, device=self.device)
        batch_data = []
        batch_locations = []

        for i in range(0, mix.shape[1], step):
            chunk = mix[:, i:i+chunk_size]
            length = chunk.shape[-1]
            if length < chunk_size: # last chunk, pad 
                pad_width = ((0,0), (0, chunk_size - length))
                if length > chunk_size // 2 + 1:
                    chunk = np.pad(chunk, pad_width, mode='reflect')
                else:
                    chunk = np.pad(chunk, pad_width, mode='constant')
            batch_data.append(torch.as_tensor(chunk, dtype=torch.float32, device=self.device))
            batch_locations.append((i, length))

            if len(batch_data) >= batch_size:
                with torch.no_grad():
                    arr = torch.stack(batch_data, axis=0)
                    arr = self.stft(arr)
                    x = self._predict_torch(arr)[0]
                    if not torch.is_tensor(x):
                        x = torch.as_tensor(x, device=self.device)
                    x = self.stft.inverse(x)
                window = windowingArray
                if i - step == 0:  # First audio chunk, no fadein
                    window = window.clone()
                    window[:fade_size] = 1
                elif i >= mix.shape[1]:  # Last audio chunk, no fadeout
                    window = window.clone()
                    window[-fade_size:] = 1

                for j in range(len(batch_locations)):
                    start, l = batch_locations[j]
                    result[..., start:start+l] += x[j][..., :l] * window[..., :l]
                    counter[..., start:start+l] += window[..., :l]

                batch_data = []
                batch_locations = []

        estimated_sources = result / counter
        estimated_sources = torch.nan_to_num(estimated_sources, nan=0.0)

        if full_length > 2 * border:
            # Remove pad
            estimated_sources = estimated_sources[..., border:-border]

        estimated_sources = estimated_sources.detach().cpu().numpy()
        return {k: v for k, v in zip(['kick', 'snare', 'toms', 'hh', 'ride', 'crash'], estimated_sources)}

    def _process(self, audio_path: Optional[str] = None, pcm: Optional[MusionPCM] = None) -> dict:
        mix = self._load_pcm(audio_path, pcm).samples[0]
        mix *= 10 ** (9 / 20) # 9dB gain

        mix = np.stack([mix, mix], axis=-1)   # Convert mono to stereo [time, channels]

        res = self.demix_track(mix.T)

        return res

    @property
    def result_keys(self):
        return ['wav']