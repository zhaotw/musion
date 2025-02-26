import os
from multiprocessing.managers import BaseManager, ValueProxy, AcquirerProxy, EventProxy
from multiprocessing import Process, Manager
import socket
import time
from functools import partial
import logging

from musion.utils.dynamic_gpu_parallel import progress_monitor, create_workers_on_one_node, get_queue

master_addr = os.getenv('MASTER_ADDR', 'localhost')
try:
    master_addr = socket.gethostbyname(master_addr)
except socket.gaierror:
    logging.warning(f"Failed to resolve hostname {master_addr}, using localhost instead.")
    master_addr = 'localhost'

DIST_INFO = {
    'master_addr': master_addr,
    'master_port': int(os.getenv('MASTER_PORT', 4513)),
    'world_size': int(os.getenv('WORLD_SIZE', 1)),
    'rank': int(os.getenv('RANK', 0))
}

AUTH_KEY = b'musion_distributed'

PATH_QUEUE = None
def get_path_queue(audio_list):
    global PATH_QUEUE
    if PATH_QUEUE is None:
        PATH_QUEUE = get_queue(audio_list, 1)
    return PATH_QUEUE

def dummy_get(instance):
    return instance

def master_process(mia_module, audio_list, num_processes_per_gpu, **kwargs):
    BaseManager.register('get_file_queue', callable=partial(get_path_queue, audio_list))
    manager = Manager()
    lock, update_event, processed = manager.Lock(), manager.Event(), manager.Value('i', 0)

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

    create_workers_on_one_node(mia_module, manager.get_file_queue(), process_state, num_processes_per_gpu, **kwargs)

    monitor.join()
    manager.shutdown()

def create_worker_node():
    BaseManager.register('get_file_queue')
    BaseManager.register('lock', proxytype=AcquirerProxy)
    BaseManager.register('update_event', proxytype=EventProxy)
    BaseManager.register('processed', proxytype=ValueProxy)

    manager = BaseManager(address=(DIST_INFO['master_addr'], DIST_INFO['master_port']),
                          authkey=AUTH_KEY
                          )

    for attempt in range(15):
        try:
            manager.connect()
            logging.info(f"Rank {DIST_INFO['rank']} connected to the server successfully on attempt {attempt}.")
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

    raise ConnectionError("Failed to connect to the server after multiple attempts.")

def worker_process(mia_module, num_processes_per_gpu, **kwargs):
    manager = create_worker_node()
    process_state = {
        "lock": manager.lock(),
        "update_event": manager.update_event(),
        "processed": manager.processed()
    }
    logging.info(f"Worker node {DIST_INFO['rank']} working...")
    create_workers_on_one_node(mia_module, manager.get_file_queue(), process_state, num_processes_per_gpu, **kwargs)

def distributed_infer(mia_module, audio_list, num_processes_per_gpu=1, **kwargs):
    if DIST_INFO['rank'] == 0:
        process = Process(target=master_process, 
                          name="server_process",
                          args=(mia_module, audio_list, num_processes_per_gpu), 
                          kwargs=kwargs)
    else:   
        process = Process(target=worker_process, 
                          name="client_process",
                          args=(mia_module, num_processes_per_gpu),
                          kwargs=kwargs)
    process.start()
    process.join()
