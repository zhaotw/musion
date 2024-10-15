from multiprocessing import Pool
from functools import partial
import threading
import math
from typing import Callable

import torch

def parallel_worker(file, task_cls, device, **kwargs):
    task_instance = task_cls(device=device)
    task_instance(file, **kwargs)

def parallel_process(num_workers: int, task_cls: Callable, file_list: list, **kwargs):
    if torch.cuda.device_count() > 1:
        num_files_per_device = math.ceil(len(file_list) / torch.cuda.device_count())
        pools = []
        for i in range(torch.cuda.device_count()):
            device = 'cuda:' + str(i)
            p = Pool(num_workers)
            p.map_async(partial(parallel_worker, task_cls=task_cls, device=device, **kwargs), 
                         file_list[i * num_files_per_device:(i + 1) * num_files_per_device])
            p.close()
            pools.append(p)
        for p in pools:
            p.join()
    else:
        with Pool(num_workers) as pool:
            pool.map(partial(parallel_worker, task_cls=task_cls, device=None, **kwargs), file_list)

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