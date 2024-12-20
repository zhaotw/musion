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
    completed_processes = 0
    num_processes = process_state["progress_dict"]["num_processes"]

    with tqdm(total=total_items, desc="Processing items", disable=False) as progress_bar:
        while completed_processes < num_processes:
            # process_state["update_event"].wait()  # 等待更新事件
            processed_now = process_state["progress_dict"]["processed"]
            progress_update = processed_now - processed_count
            if progress_update > 0:
                progress_bar.update(progress_update)
                processed_count = processed_now
            process_state["update_event"].clear()  # reset update event, for next update

            if process_state["complete_event"].is_set():
                completed_processes += 1
                process_state["complete_event"].clear()  # reset complete event, for next check

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
        except Exception as e:
            logging.debug(f"Error processing {audio_path}: {e}")
            torch.cuda.empty_cache()
        finally:
            with process_state["lock"]:
                process_state["progress_dict"]["processed"] += 1
                process_state["update_event"].set()  # trigger update progress bar
    process_state["complete_event"].set()

def dynamic_gpu_parallel_infer(mia_module, audio_list, num_processes_per_gpu=1, **kwargs):
    """
    Dynamic GPU task allocation, file-level parallel inference, which can alleviate the problem of unbalanced GPU load.
    Also monitor the global processing progress.
    Similar to DDP, each GPU starts num_processes_per_gpu processes
    """
    file_queue = get_queue(audio_list, 1)

    process_state = {
        "progress_dict": Manager().dict(),
        "update_event": Event(),
        "complete_event": Event(),
        "lock": Lock(),
    }
    progress_dict = process_state["progress_dict"]
    progress_dict["processed"] = 0

    available_gpus = list(range(torch.cuda.device_count()))
    logging.info(f"Available GPUs: {available_gpus}")
    progress_dict["num_processes"] = len(available_gpus) * num_processes_per_gpu

    processes = []
    for i in available_gpus:
        for j in range(num_processes_per_gpu):
            p = Process(
                name=f"worker GPU-{i}-{j}",
                target=gpu_worker,
                args=(mia_module, i, file_queue, process_state),
                kwargs=kwargs,
            )
            p.start()
            processes.append(p)

    monitor = Process(target=progress_monitor, args=(len(audio_list), process_state),
                      name="progress monitor")
    monitor.start()

    for p in processes:
        p.join()

    monitor.join()
