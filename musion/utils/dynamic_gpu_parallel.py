import logging

import torch
from torch.multiprocessing import Event, Manager, Lock, Process, Queue
from tqdm import tqdm


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
            update_event.wait()  # 等待更新事件
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

    while not file_queue.empty():
        audio_path = file_queue.get()
        
        try:
            musion_task(
                audio_path=audio_path,
                **musion_kwargs['call_kwargs']
        )
        except PermissionError as e:
            print(f"PermissionError: {e}")
            exit()
        except Exception as e:
            logging.debug(f"Error processing {audio_path}: {e}")
            torch.cuda.empty_cache()

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
