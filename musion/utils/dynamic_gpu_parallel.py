import logging
import os
import socket
import traceback

import torch
from torch.multiprocessing import Event, Manager, Lock, Process, Queue
from tqdm import tqdm

master_addr = os.getenv('MASTER_ADDR', 'localhost')
try:
    master_addr = socket.gethostbyname(master_addr)
except socket.gaierror:
    logging.warning(f"Failed to resolve hostname {master_addr}, using localhost instead.")
    master_addr = 'localhost'

DIST_INFO = {
    'master_addr': master_addr,
    'master_port': int(os.getenv('MASTER_PORT', 4516)),
    'world_size': int(os.getenv('WORLD_SIZE', 1)),
    'rank': int(os.getenv('RANK', 0))
}


def get_queue(song_list, split):
    file_queue = Queue()
    for i in range(0, len(song_list), split):
        file_queue.put(song_list[i : i + split] if split > 1 else song_list[i])

    return file_queue

def progress_monitor(total_items, process_state):
    processed_count = 0

    update_event = process_state["update_event"]

    with tqdm(total=total_items, desc="Processing items", disable=False) as progress_bar:
        while processed_count < total_items:
            if not update_event.wait(timeout=1000):
                logging.warning(f"Monitor process timeout")
                return
            processed_now = process_state["processed"].value
            progress_update = processed_now - processed_count
            if progress_update > 0:
                progress_bar.update(progress_update)
                processed_count = processed_now
            update_event.clear()  # reset update event, for next update

def gpu_worker(
    task_name, gpu_id: int, file_queue: Queue, process_state,
    **musion_kwargs
):
    musion_task = task_name(device=f"cuda:{gpu_id}", **musion_kwargs['init_kwargs'])

    timeout = 60 if DIST_INFO['rank'] > 0 else 5

    while True:
        try:
            audio_path = file_queue.get(timeout=timeout)
        except:
            logging.warning(f"Process timeout")
            return

        try:
            musion_task(
                audio_path=audio_path,
                **musion_kwargs['call_kwargs']
            )
        except PermissionError as e:
            logging.error(f"PermissionError: {e}")
            exit()
        except torch.cuda.OutOfMemoryError as e:
            logging.warning(f"OOM, will requeue the task and retry later")
            torch.cuda.empty_cache()
            try:
                file_queue.put(audio_path)
            except Exception:
                logging.warning(f"Failed to requeue {audio_path}, server might be closed")
            continue
        except Exception as e:
            logging.error(f"Error processing {audio_path}: {e}")
            traceback.print_exc()
            exit(1)

        with process_state["lock"]:
            process_state["processed"].value += 1
            process_state["update_event"].set()  # trigger update progress bar

def create_workers_on_one_node(musion_module, file_queue, process_state, num_processes_per_gpu=1, **kwargs):
    available_gpus = list(range(torch.cuda.device_count()))
    logging.info(f"Available GPUs: {available_gpus}")
    processes = []

    for i in available_gpus:
        for j in range(num_processes_per_gpu):
            p = Process(
                name=f"worker GPU-{i}-{j}",
                target=gpu_worker,
                args=(musion_module, i, file_queue, process_state),
                kwargs=kwargs,
            )
            p.start()
            processes.append(p)
    for p in processes:
        p.join()

def dynamic_gpu_parallel_infer(musion_module, audio_list, num_processes_per_gpu=1, **kwargs):
    """
    Dynamic GPU task allocation, file-level parallel inference, which can alleviate the problem of unbalanced GPU load.
    Also monitor the global processing progress.
    Similar to DDP, each GPU starts num_processes_per_gpu processes
    """
    file_queue = get_queue(audio_list, 1)

    process_state = {
        "lock": Lock(),
        "update_event": Event(),
        "processed": Manager().Value('i', 0),
    }

    monitor = Process(target=progress_monitor, args=(len(audio_list), process_state),
                      name="progress monitor")
    monitor.start()

    create_workers_on_one_node(musion_module, file_queue, process_state, num_processes_per_gpu, **kwargs)

    monitor.join()
