from multiprocessing import Pool
from functools import partial
import threading
import math
from typing import Callable

def parallel_worker(file, task_cls, **kwargs):
    task_instance = task_cls()
    task_instance(file, **kwargs)

def parallel_process(num_workers: int, task_cls: Callable, file_list: list, **kwargs):
    with Pool(num_workers) as pool:
        pool.map(partial(parallel_worker, task_cls=task_cls, **kwargs), file_list)

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