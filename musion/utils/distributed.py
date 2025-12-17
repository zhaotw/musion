from multiprocessing.managers import BaseManager, ValueProxy, AcquirerProxy, EventProxy
from multiprocessing import Process, Manager
import time
from functools import partial
import logging

from musion.utils.dynamic_gpu_parallel import progress_monitor, create_workers_on_one_node, DIST_INFO


AUTH_KEY = b'musion_distributed'


def dummy_get(instance):
    return instance

def master_process(musion_module, audio_list, num_processes_per_gpu, **kwargs):
    sr_manager = Manager()
    path_queue = sr_manager.Queue()
    for i in audio_list:
        path_queue.put(i)
    lock, update_event, processed = sr_manager.Lock(), sr_manager.Event(), sr_manager.Value('i', 0)

    BaseManager.register('path_queue', callable=partial(dummy_get, path_queue))
    BaseManager.register('lock', partial(dummy_get, lock))
    BaseManager.register('update_event', partial(dummy_get, update_event))
    BaseManager.register('processed', partial(dummy_get, processed), proxytype=ValueProxy)

    manager = BaseManager(address=('', DIST_INFO['master_port']),
                          authkey=AUTH_KEY
                          )

    manager.start()
    logging.info(f"Master node started on {DIST_INFO['master_addr']}:{DIST_INFO['master_port']}")

    process_state = {
        "lock": lock,
        "update_event": update_event,
        "processed": processed
    }

    monitor = Process(target=progress_monitor,
                      args=(len(audio_list), process_state),
                      name="progress monitor")
    monitor.start()
    logging.info("Progress monitor process started")

    create_workers_on_one_node(musion_module, manager.path_queue(), process_state, num_processes_per_gpu, **kwargs)

    monitor.join()
    manager.shutdown()

def create_worker_node():
    BaseManager.register('path_queue')
    BaseManager.register('lock', proxytype=AcquirerProxy)
    BaseManager.register('update_event', proxytype=EventProxy)
    BaseManager.register('processed', proxytype=ValueProxy)

    manager = BaseManager(address=(DIST_INFO['master_addr'], DIST_INFO['master_port']),
                          authkey=AUTH_KEY
                          )

    for attempt in range(15):
        try:
            manager.connect()
            logging.info(f"Rank {DIST_INFO['rank']} connected to the server {DIST_INFO['master_addr']}:{DIST_INFO['master_port']} successfully on attempt {attempt}.")
            return manager
        except ConnectionRefusedError:
            time.sleep(10)
        finally:
            if manager._state.value == 1: # fast test for reading shared state
                with manager.lock():
                    update_event = manager.update_event()
                    update_event.set()
                    processed = manager.processed()
                    logging.debug(f'RANK {DIST_INFO["rank"]} init processed: {processed.value}')

    raise ConnectionError(f"{DIST_INFO['rank']} failed to connect to the server after multiple attempts.")

def worker_process(musion_module, num_processes_per_gpu, **kwargs):
    manager = create_worker_node()
    process_state = {
        "lock": manager.lock(),
        "update_event": manager.update_event(),
        "processed": manager.processed()
    }
    logging.info(f"Worker node {DIST_INFO['rank']} working...")
    create_workers_on_one_node(musion_module, manager.path_queue(), process_state, num_processes_per_gpu, **kwargs)

def distributed_infer(musion_module, audio_list, num_processes_per_gpu=1, **kwargs):
    if DIST_INFO['rank'] == 0:
        process = Process(target=master_process, 
                          name="server_process",
                          args=(musion_module, audio_list, num_processes_per_gpu), 
                          kwargs=kwargs)
    else:   
        process = Process(target=worker_process, 
                          name="client_process",
                          args=(musion_module, num_processes_per_gpu),
                          kwargs=kwargs)
    process.start()
    process.join()
