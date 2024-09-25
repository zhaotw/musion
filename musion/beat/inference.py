import os
from typing import Any

import torch
from torchaudio.transforms import MelSpectrogram

from musion.separate.utils import *
from musion.util.base import FeatConfig, MusionBase
from musion.beat.postprocessor import Postprocessor
from musion.beat.utils import *


MODULE_PATH = os.path.dirname(__file__)


class Beat(MusionBase):
    def __init__(self, meter_change_detection: bool = False) -> None:
        super().__init__(True, 
                         MelSpectrogram(**self._feat_cfg.__dict__),
                         os.path.join(MODULE_PATH, 'beat.onnx'))

        self.postprocessor = Postprocessor('minimal' if meter_change_detection else 'dbn')

    @property
    def _feat_cfg(self) -> FeatConfig:
        return FeatConfig(
            sample_rate=22050,
            n_fft=1024,
            hop_length=441,
            f_min=30, 
            f_max=11000, 
            n_mels=128, 
            mel_scale='slaney',
            power=1,
            normalized='frame_length'
        )

    def _process(self, samples: np.ndarray, **kwargs: Any) -> dict:
        samples = torch.from_numpy(samples[0]).to(self.device)
        feat = self._feat(samples).T
        spec = torch.log1p(1000 * feat)

        beat_logits, downbeat_logits = self.split_predict_aggregate(
                    spect=spec,
                    chunk_size=1500,
                    overlap_mode="keep_first",
                    border_size=6,
                )
        res = self.postprocessor(beat_logits, downbeat_logits)
        res = convert_to_std_result(*res)

        return {"beats": res}

    def split_predict_aggregate(
        self, 
        spect: torch.Tensor,
        chunk_size: int,
        border_size: int,
        overlap_mode: str,
    ) -> dict:
        """
        Function for pieces that are longer than the training length of the model.
        Split the input piece into chunks, run the model on them, and aggregate the predictions.
        The spect is supposed to be a torch tensor of shape (time x bins), i.e., unbatched, and the output is also unbatched.

        Args:
            spect (torch.Tensor): the input piece
            chunk_size (int): the length of the chunks
            border_size (int): the size of the border that is discarded from the predictions
            overlap_mode (str): how to handle overlaps between chunks

        Returns:
            dict: the model framewise predictions for the hole piece as a dictionary containing 'beat' and 'downbeat' predictions.
        """
        # split the piece into chunks
        chunks, starts = split_piece(
            spect, chunk_size, border_size=border_size, avoid_short_end=True
        )

        # run the model
        pred_chunks = [self._predict(chunk[None].cpu().numpy()) for chunk in chunks]
        # remove the extra dimension in beat and downbeat prediction due to batch size 1
        pred_chunks = [
            {"beat": torch.from_numpy(p[0][0]), "downbeat": torch.from_numpy(p[1][0])} for p in pred_chunks
        ]
        piece_prediction_beat, piece_prediction_downbeat = aggregate_prediction(
            pred_chunks,
            starts,
            spect.shape[0],
            chunk_size,
            border_size,
            overlap_mode,
            spect.device,
        )

        return piece_prediction_beat, piece_prediction_downbeat

    @property
    @staticmethod
    def result_keys(self) -> list[str]:
        return ["beats"]
