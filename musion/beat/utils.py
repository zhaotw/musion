import torch
import torch.nn.functional as F
import numpy as np
from itertools import chain

def zeropad(spect: torch.Tensor, left: int = 0, right: int = 0):
    """
    Pads a tensor spectrogram matrix of shape (time x bins) with `left` frames in the beginning and `right` frames in the end.
    """
    if left == 0 and right == 0:
        return spect
    else:
        return F.pad(spect, (0, 0, left, right), "constant", 0)

def split_piece(
    spect: torch.Tensor,
    chunk_size: int,
    border_size: int = 6,
    avoid_short_end: bool = True,
):
    """
    Split a tensor spectrogram matrix of shape (time x bins) into time chunks of `chunk_size` and return the chunks and starting positions.
    The `border_size` is the number of frames assumed to be discarded in the predictions on either side (since the model was not trained on the input edges due to the max-pool in the loss).
    To cater for this, the first and last chunk are padded by `border_size` on the beginning and end, respectively, and consecutive chunks overlap by `border_size`.
    If `avoid_short_end` is true, the last chunk start is shifted left to ends at the end of the piece, therefore the last chunk can potentially overlap with previous chunks more than border_size, otherwise it will be a shorter segment.
    If the piece is shorter than `chunk_size`, avoid_short_end is ignored and the piece is returned as a single shorter chunk.

    Args:
        spect (torch.Tensor): The input spectrogram tensor of shape (time x bins).
        chunk_size (int): The size of the chunks to produce.
        border_size (int, optional): The size of the border to overlap between chunks. Defaults to 6.
        avoid_short_end (bool, optional): If True, the last chunk is shifted left to end at the end of the piece. Defaults to True.
    """
    # generate the start and end indices
    starts = np.arange(
        -border_size, len(spect) - border_size, chunk_size - 2 * border_size
    )
    if len(spect) <= chunk_size - 2 * border_size:
        starts = np.array([-border_size])
        chunks = [zeropad(spect, left=border_size, right=chunk_size - len(spect) - border_size)]
        return chunks, starts

    if avoid_short_end and len(spect) > chunk_size - 2 * border_size:
        # if we avoid short ends, move the last index to the end of the piece - (chunk_size - border_size)
        starts[-1] = len(spect) - (chunk_size - border_size)
    # generate the chunks
    chunks = [
        zeropad(
            spect[max(start, 0) : min(start + chunk_size, len(spect))],
            left=max(0, -start),
            right=max(0, min(border_size, start + chunk_size - len(spect))),
        )
        for start in starts
    ]
    return chunks, starts

def aggregate_prediction(
    pred_chunks: list,
    starts: list,
    full_size: int,
    chunk_size: int,
    border_size: int,
    overlap_mode: str,
    device: str | torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Aggregates the predictions for the whole piece based on the given prediction chunks.

    Args:
        pred_chunks (list): List of prediction chunks, where each chunk is a dictionary containing 'beat' and 'downbeat' predictions.
        starts (list): List of start positions for each prediction chunk.
        full_size (int): Size of the full piece.
        chunk_size (int): Size of each prediction chunk.
        border_size (int): Size of the border to be discarded from each prediction chunk.
        overlap_mode (str): Mode for handling overlapping predictions. Can be 'keep_first' or 'keep_last'.
        device (torch.device): Device to be used for the predictions.

    Returns:
        tuple: A tuple containing the aggregated beat predictions and downbeat predictions as torch tensors for the whole piece.
    """
    if border_size > 0:
        # cut the predictions to discard the border
        pred_chunks = [
            {
                "beat": pchunk["beat"][border_size:-border_size],
                "downbeat": pchunk["downbeat"][border_size:-border_size],
            }
            for pchunk in pred_chunks
        ]
    # aggregate the predictions for the whole piece
    piece_prediction_beat = torch.full((full_size,), -1000.0, device=device)
    piece_prediction_downbeat = torch.full((full_size,), -1000.0, device=device)
    if overlap_mode == "keep_first":
        # process in reverse order, so predictions of earlier excerpts overwrite later ones
        pred_chunks = reversed(list(pred_chunks))
        starts = reversed(list(starts))
    valid_size = min(full_size, chunk_size - 2 * border_size)
    for start, pchunk in zip(starts, pred_chunks):
        piece_prediction_beat[
            start + border_size : start + chunk_size - border_size
        ] = pchunk["beat"][:valid_size]
        piece_prediction_downbeat[
            start + border_size : start + chunk_size - border_size
        ] = pchunk["downbeat"][:valid_size]
    return piece_prediction_beat, piece_prediction_downbeat

def convert_to_std_result(beats: np.ndarray, downbeats: np.ndarray) -> None:
    """
    Save beat information to a tab-separated file in the standard .beats format:
    each line has a time in seconds, a tab, and a beat number (1 = downbeat).
    The function requires that all downbeats are also listed as beats.

    Args:
        beats (numpy.ndarray): Array of beat positions in seconds (including downbeats).
        downbeats (numpy.ndarray): Array of downbeat positions in seconds.

    Returns:
        None
    """
    # check if all downbeats are beats
    if not np.all(np.isin(downbeats, beats)):
        raise ValueError("Not all downbeats are beats.")

    # handle pickup measure, by considering the beat count of the first full measure
    if len(downbeats) >= 2:
        # find the number of beats between the first two downbeats
        first_downbeat, second_downbeat = np.searchsorted(beats, downbeats[:2])
        beats_in_first_measure = second_downbeat - first_downbeat
        # find the number of beats before the first downbeat
        pickup_beats = first_downbeat
        # derive where to start counting
        if pickup_beats < beats_in_first_measure:
            start_counter = beats_in_first_measure - pickup_beats
        else:
            print(
                "WARNING: There are more beats in the pickup measure than in the first measure. The beat count will start from 2 without trying to estimate the length of the pickup measure."
            )
            start_counter = 1
    else:
        print(
            "WARNING: There are less than two downbeats in the predictions. Something may be wrong. The beat count will start from 2 without trying to estimate the length of the pickup measure."
        )
        start_counter = 1

    counter = start_counter
    downbeats = chain(downbeats, [-1])
    next_downbeat = next(downbeats)

    res = []
    for beat in beats:
        if beat == next_downbeat:
            counter = 1
            next_downbeat = next(downbeats)
        else:
            counter += 1
        res.append(f"{beat:.2f}\t{counter:d}")

    return res
