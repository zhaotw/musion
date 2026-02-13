import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

def makeFrame(x, hopSize, windowSize):
    nFrame = math.ceil((x.shape[-1])/hopSize)+1

    lPad = windowSize//2
    rPad = (nFrame-1)*hopSize+windowSize//2 - x.shape[-1]

    x = F.pad(x, (lPad, rPad))

    frames = x.unfold(-1, windowSize, hopSize)
    
    return frames

class MelSpectrum(nn.Module):
    def __init__(self, n_fft, f_min, f_max, n_mels, sample_rate, **kwargs):
        super().__init__()

        self.eps = 1e-5
        self.mel_scale = torchaudio.transforms.MelScale(n_mels, sample_rate, f_min, f_max, n_fft//2+1)

        hann_win = torch.hann_window(n_fft)

        sigma = torch.tensor([-1.1706887483596802, -0.8110867142677307,  0.6248952746391296,
            -0.6810300946235657, -1.2125477790832520])
        center = torch.tensor([-0.6510074734687805, -0.2153073400259018, -0.0315931625664234,
            0.3345024585723877,  0.9446282386779785])

        sigma = torch.sigmoid(sigma)
        center = torch.sigmoid(center)

        x = torch.arange(n_fft)
        g_win = (-0.5* ((x.unsqueeze(1) - n_fft*center)/(sigma*n_fft/2))**2).exp()

        self.wins = torch.cat([hann_win.unsqueeze(0), g_win.t()], dim = 0)

    def forward(self, frames):
        # output format: (.,  #frame, #freqBin, #featureChannel)
        spectrogram = torch.fft.rfft(
                (frames.unsqueeze(-2) * self.wins.to(frames.device)),
                norm= "ortho").transpose(-1,-2)
        spectrogram = (spectrogram).abs().pow(2)
        spectrogram = spectrogram.mean(dim = -4, keepdim = True)
        mel = self.mel_scale(spectrogram)

        # with normalization
        eps = self.eps
        return torch.log1p(mel / eps) / (-math.log(eps))
