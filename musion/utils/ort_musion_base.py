from typing import Optional, List, Union, Callable
import os
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from multiprocessing import cpu_count

import onnxruntime as ort
import numpy as np

from musion.utils.base import MusionBase, MusionPCM

class OrtMusionBase(MusionBase):
    def __init__(self, model_path: str, device: Optional[str] = None) -> None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file: {model_path} not found. Download it first following steps in README.")

        super().__init__(device)

        ort_options = ort.SessionOptions()
        ort_options.enable_cpu_mem_arena = False
        ort_options.enable_mem_pattern = False
        providers=[]
        if 'cuda' in self.device:
            device_id = 0
            if 'cuda:' in self.device:
                device_id = int(self.device.split(':')[1])
            providers.append(('CUDAExecutionProvider', {'device_id': device_id}))
        providers.append('CPUExecutionProvider')
        self.__ort_session = ort.InferenceSession(model_path, sess_options=ort_options, providers=providers)

        num_threads = max(cpu_count() // 2, 1)
        self.__batch_predict_pool = ThreadPoolExecutor(num_threads)

    def _load_pcm(self, audio_path: Optional[str] = None, pcm: Optional[MusionPCM] = None) -> MusionPCM:
        pcm = super()._load_pcm(audio_path, pcm)
        pcm.samples = pcm.samples.numpy()
        return pcm

    def _batch_process(self, fn: Callable, inputs: list, batch_size = 1) -> list:
        futures = []
        for i in range(0, len(inputs), batch_size):
            input = np.stack(inputs[i : i + batch_size])
            futures.append(self.__batch_predict_pool.submit(fn, input))

        res = []
        for i, f in enumerate(futures):
            res += f.result()

        return res

    def _predict(self, model_input: Union[List[np.ndarray], np.ndarray]):
        if isinstance(model_input, np.ndarray):
            model_input = [model_input]
        ort_input = {i.name: d.astype(np.float32) for i, d in zip(self.__ort_session.get_inputs(), model_input)}
        return self.__ort_session.run(None, input_feed=ort_input)