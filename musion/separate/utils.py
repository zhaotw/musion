import typing as tp

import torch as th
from torch.nn import functional as F

def spectro(x, n_fft, hop_length):
    *other, length = x.shape
    x = x.reshape(-1, length)
    is_mps = x.device.type == 'mps'
    if is_mps:
        x = x.cpu()
    z = th.stft(x,
                n_fft,
                hop_length,
                window=th.hann_window(n_fft).to(x),
                win_length=n_fft,
                normalized=True,
                center=True,
                return_complex=True,
                pad_mode='reflect')
    _, freqs, frame = z.shape
    return z.view(*other, freqs, frame)

def ispectro(z, hop_length=None, length=None, pad=0):
    *other, freqs, frames = z.shape
    n_fft = 2 * freqs - 2
    z = z.view(-1, freqs, frames)
    win_length = n_fft // (1 + pad)
    is_mps = z.device.type == 'mps'
    if is_mps:
        z = z.cpu()
    x = th.istft(z,
                 n_fft,
                 hop_length,
                 window=th.hann_window(win_length).to(z.real),
                 win_length=win_length,
                 normalized=True,
                 length=length,
                 center=True)
    _, length = x.shape
    return x.view(*other, length)

def pad1d(x: th.Tensor, paddings: tp.Tuple[int, int], mode: str = 'constant', value: float = 0.):
    """Tiny wrapper around F.pad, just to allow for reflect padding on small input.
    If this is the case, we insert extra 0 padding to the right before the reflection happen."""
    x0 = x
    length = x.shape[-1]
    padding_left, padding_right = paddings
    if mode == 'reflect':
        max_pad = max(padding_left, padding_right)
        if length <= max_pad:
            extra_pad = max_pad - length + 1
            extra_pad_right = min(padding_right, extra_pad)
            extra_pad_left = extra_pad - extra_pad_right
            paddings = (padding_left - extra_pad_left, padding_right - extra_pad_right)
            x = F.pad(x, (extra_pad_left, extra_pad_right))
    out = F.pad(x, paddings, mode, value)

    return out

def center_trim(tensor, reference: int):
    """
    Center trim `tensor` with respect to `reference`, along the last dimension.
    `reference` can also be a number, representing the length to trim to.
    If the size difference != 0 mod 2, the extra sample is removed on the right side.
    """
    delta = tensor.shape[-1] - reference

    if delta:
        tensor = tensor[..., delta // 2:-(delta - delta // 2)]
    return tensor

class TensorChunk:
    def __init__(self, tensor, offset, length):
        total_length = tensor.shape[-1]


        length = min(total_length - offset, length)


        self.tensor = tensor
        self.offset = offset

        self.length = length

    def padded(self, target_length):
        delta = target_length - self.length
        total_length = self.tensor.shape[-1]

        start = self.offset - delta // 2
        end = start + target_length

        correct_start = max(0, start)
        correct_end = min(total_length, end)

        pad_left = correct_start - start
        pad_right = end - correct_end

        out = F.pad(self.tensor[..., correct_start:correct_end], (pad_left, pad_right))

        return out

def magnitude(z):
    # return the magnitude of the spectrogram, except when cac is True,
    # in which case we just move the complex dimension to the channel one.

    B, C, Fr, T = z.shape
    m = th.view_as_real(z).permute(0, 1, 4, 2, 3)
    m = m.reshape(B, C * 2, Fr, T)
    return m

def mask(m):
    B, S, C, Fr, T = m.shape
    out = m.view(B, S, -1, 2, Fr, T).permute(0, 1, 2, 4, 5, 3)
    out = th.view_as_complex(out.contiguous())
    return out