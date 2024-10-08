from typing import Callable
import os
import threading
import math
import copy

import numpy as np

def get_file_list(dir_path):
    return [os.path.join(dir_path, f) for f in os.listdir(dir_path)]

def get_file_name(file_path):
    """ /dir/test.wav -> test """
    return os.path.splitext(os.path.split(file_path)[1])[0]

def check_exist(audio_path, save_cfg):
    audio_name = get_file_name(audio_path)
    for key in save_cfg.keys:
        tgt_path = os.path.join(save_cfg.dir_path, f"{audio_name}.{key}")
        if not os.path.exists(tgt_path):
            return False

    return True

def parallel_process(num_threads: int, fn: Callable, file_list: list, *fn_args):
    num_threads = min(num_threads, len(file_list))
    files_per_thread = math.ceil(len(file_list) / num_threads)
    threads = []

    for i in range(0, len(file_list), files_per_thread):
        end_idx = min(i + files_per_thread, len(file_list))
        threads.append(threading.Thread(target=fn, args=[file_list[i:end_idx], *fn_args]))
        threads[-1].start()
    for th in threads:
        th.join()

def normalize(data):
    new_np = np.zeros(len(data))
    max_score = np.max(data)
    min_score = np.min(data)
    overlap = max_score - min_score
    for i in range(len(data)):
        new_np[i] = (data[i] - min_score) / overlap
    return new_np

def median_filter(result_np, sample):
    begin = side = int(sample // 2)
    for i in range(begin):
        result_np = np.insert(result_np, 0, result_np[2 * i])
        result_np = np.append(result_np, result_np[-(2 * i)])

    filtered_np = copy.deepcopy(result_np)

    for i in range(begin, (len(filtered_np) - side)):
        group_s = i - side
        group_e = i + side + 1
        window = result_np[group_s: group_e]

        r_max = np.max(window)
        r_min = np.min(window)

        mid_value = float((sum(window) - r_min - r_max) / (len(window) - 2))
        filtered_np[i] = mid_value
    for i in range(begin):
        filtered_np = np.delete(filtered_np, 0)
        filtered_np = np.delete(filtered_np, len(filtered_np) - 1)
    return filtered_np

def enframe(x, hop, frame_size, window_start=0):
    """Enframe long sequence to short segments.

    Args:
        x: (audio_samples)
        frame_size: int

    Returns:
        batch: (N, frame_size)
    """
    #TODO np.pad may be slow, consider np.zeros_like

    batch = []

    pointer = 0 - window_start
    while pointer + frame_size <= x.shape[0]:
        if pointer < 0:
            batch.append(np.pad(x[0 : pointer + frame_size], (-pointer, 0), 'constant',
                   constant_values=(0, 0)))
        else:
            batch.append(x[pointer : pointer + frame_size])
        pointer += hop

    # pad
    if batch[-1].shape[0] < frame_size:
        batch[-1] = np.pad(batch[-1], (0, frame_size - batch[-1].shape[0]), 'constant',
               constant_values=(0, 0))

    return batch

def deframe(x):
    """Deframe predicted segments to original sequence.

    Args:
        x: (N, segment_frames, classes_num)

    Returns:
        y: (audio_frames, classes_num)
    """
    if x.shape[0] == 1:
        return x[0]
    else:
        x = x[:, 0 : -1, :]
        # Remove an extra frame in the end of each segment caused by the
        # 'center=True' argument when calculating spectrogram.
        (N, segment_samples, classes_num) = x.shape
        assert segment_samples % 4 == 0

        y = []
        y.append(x[0, 0 : int(segment_samples * 0.75)])
        for i in range(1, N - 1):
            y.append(x[i, int(segment_samples * 0.25) : int(segment_samples * 0.75)])
        y.append(x[-1, int(segment_samples * 0.25) :])
        y = np.concatenate(y, axis=0)

        return y

def convert_audio_channels(wav: np.ndarray, channels: int):
    """Convert audio to the given number of channels."""
    src_channels = wav.shape[-2]
    if src_channels == channels:
        pass
    elif channels == 1:
        # Case 1:
        # The caller asked 1-channel audio, but the stream have multiple
        # channels, downmix all channels.
        wav = wav.mean(-2, keepdims=True)
    elif src_channels == 1:
        # Case 2:
        # The caller asked for multiple channels, but the input file have
        # one single channel, replicate the audio over all channels.
        wav = np.tile(wav, (channels, 1))
    elif src_channels >= channels:
        # Case 3:
        # The caller asked for multiple channels, and the input file have
        # more channels than requested. In that case return the first channels.
        wav = wav[..., :channels, :]
    else:
        # Case 4: What is a reasonable choice here?
        raise ValueError('The audio file has less channels than requested but is not mono.')
    return wav
