from functools import partial
import threading
import math
from typing import Callable
import logging

import torch
import torch.multiprocessing as mp
from tqdm import tqdm

from musion.utils.dynamic_gpu_parallel import dynamic_gpu_parallel_infer
from musion.utils.tools import get_file_name
from musion.utils.distributed import distributed_infer, DIST_INFO

def parallel_worker(file, task_cls, **kwargs):
    task_instance = task_cls(**kwargs['init_kwargs'])
    res = task_instance(file, **kwargs['call_kwargs'])
    stem = get_file_name(file)
    return stem, res

def multiprocess_with_tqdm(func, iterable, num_workers=mp.cpu_count()//2, return_dict=True):
    logging.info(f"Using {num_workers} workers for multiprocessing")
    res_dict = {}
    with tqdm(total=len(iterable), unit="piece") as pbar:
        with mp.Pool(num_workers) as p:
            for res in p.imap_unordered(func, iterable):
                pbar.update(1)
                if return_dict:
                    res_dict[res[0]] = res[1]

    return res_dict if return_dict else None

def parallel_process(num_workers: int, task_cls: Callable, file_list: list, **kwargs):
    device = kwargs['init_kwargs'].get('device', None)

    if DIST_INFO['world_size'] > 1:
        return distributed_infer(task_cls, file_list, num_workers, **kwargs)

    if torch.cuda.device_count() > 1 and \
        (not device or 'cuda' in device):
        return dynamic_gpu_parallel_infer(task_cls, file_list, num_workers, **kwargs)
    else:
        return multiprocess_with_tqdm(
            partial(parallel_worker, task_cls=task_cls, **kwargs),
            file_list,
            num_workers
        )

def concurrent_process(num_threads: int, fn: Callable, file_list: list, *fn_args):
    num_threads = min(num_threads, len(file_list))
    files_per_thread = math.ceil(len(file_list) / num_threads)
    threads = []

    for i in range(0, len(file_list), files_per_thread):
        end_idx = min(i + files_per_thread, len(file_list))
        threads.append(threading.Thread(target=fn, args=[file_list[i:end_idx], *fn_args]))
        threads[-1].start()
    for th in threads:
        th.join()
