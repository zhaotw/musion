from typing import Any, Optional, List, Union, Callable
import os
import dataclasses
import logging
import abc
import time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import librosa
import torchaudio
import torch
import onnxruntime as ort
from tqdm import tqdm

from musion.utils.tools import *
from musion.utils.parallel import *

logging.basicConfig(level=logging.WARNING)

@dataclasses.dataclass
class MusionPCM:
    samples: np.ndarray
    sample_rate: int

@dataclasses.dataclass
class SaveConfig:
    dir_path: str
    keys: List[str] = None

@dataclasses.dataclass
class FeatConfig:
    mono: bool
    sample_rate: int
    n_fft: Optional[int] = None
    hop_length: Optional[int] = None
    f_min: Optional[float] = None
    f_max: Optional[float] = None
    n_mels: Optional[int] = None
    normalized: Union[str, bool] = False
    power: Optional[float] = 2.0
    norm: Optional[str] = None
    mel_scale: Optional[str] = 'htk'

    @property
    def fps(self):  # frames per second
        return self.sample_rate / self.hop_length

class TaskDispatcher(metaclass=abc.ABCMeta):
    def __init__(self, musion_class, **init_kwargs) -> None:
        self.__task_class = musion_class
        self.__init_kwargs = init_kwargs
        self.__task = None
        
    @property
    def result_keys(self) -> List[str]:
        return self.get_task().result_keys

    def get_task(self):
        if self.__task is None:
            self.__task = self.__task_class(**self.__init_kwargs)
        return self.__task

    def __call__(self, audio_path: Optional[Union[List[str], str]] = None, pcm: Optional[MusionPCM] = None,
                 save_cfg: Optional[SaveConfig] = None, overwrite: bool = False, 
                 num_workers: int = 0, num_threads: int = 0, **kwargs: Any) -> dict:
        """
        num_workers: number of workers for parallel processing
        num_threads: number of threads for concurrent processing
        """
        if isinstance(audio_path, List):
            res = self.__process_multi_file(audio_path, num_workers, num_threads, save_cfg=save_cfg, overwrite=overwrite)
        elif isinstance(audio_path, str) and os.path.isdir(audio_path):
            res = self.__process_multi_file(get_file_list(audio_path), num_workers, num_threads, save_cfg=save_cfg, overwrite=overwrite)
        else:
            res = self.get_task()(audio_path, pcm, save_cfg, overwrite)

        return res

    def __process_multi_file(self, file_list: list, num_workers: int, num_threads: int, **call_kwargs) -> dict:
        res = {}

        if num_workers > 0:
            kwargs = {'init_kwargs': self.__init_kwargs, 'call_kwargs': call_kwargs}
            res = parallel_process(num_workers, self.__task_class, file_list, **kwargs)
        else:
            def serial_process(file_list, res):
                task = self.get_task()
                for audio_path in tqdm(file_list, desc=f"Processing..."):
                    cur_res = task(audio_path, **call_kwargs)
                    res[get_file_name(audio_path)] = cur_res

            if num_threads == 0:
                serial_process(file_list, res)
            else:
                concurrent_process(num_threads, serial_process, file_list, res)

        return res

class MusionBase(metaclass=abc.ABCMeta):
    def __init__(self, device: Optional[str] = None) -> None:
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        num_threads = max(cpu_count() // 2, 1)
        self.__batch_predict_pool = ThreadPoolExecutor(num_threads)

    @property
    @abc.abstractmethod
    def _feat_cfg(self) -> FeatConfig:
        raise NotImplementedError("Subclass must implement this method.")

    @abc.abstractmethod
    def _process(self, audio_path: Optional[str] = None, pcm: Optional[MusionPCM] = None) -> dict:
        raise NotImplementedError("Subclass must implement this method.")

    @property
    @abc.abstractmethod
    def result_keys(self) -> List[str]:
        raise NotImplementedError("Subclass must implement this method.")

    def _load_pcm(self, audio_path: Optional[str] = None, pcm: Optional[MusionPCM] = None) -> MusionPCM:
        if audio_path is None and pcm is None:
            raise ValueError('Should provide either audio path or pcm to proceed.')

        if audio_path and pcm:
            logging.warning('Both audio path and pcm provided, will use audio path.')
        if audio_path:
            samples, sr = librosa.load(audio_path, sr=None, mono=self._feat_cfg.mono)
            pcm = MusionPCM(samples, sr)

        if pcm.sample_rate != self._feat_cfg.sample_rate:
            pcm.samples = torchaudio.transforms.Resample(pcm.sample_rate, self._feat_cfg.sample_rate)(
                torch.from_numpy(pcm.samples))
            pcm.sample_rate = self._feat_cfg.sample_rate

        pcm.samples = np.asarray(pcm.samples)

        if pcm.samples.ndim == 1:
            pcm.samples = np.expand_dims(pcm.samples, 0)

        pcm.samples = convert_audio_channels(pcm.samples, 1 if self._feat_cfg.mono else 2)

        return pcm

    def _batch_process(self, fn: Callable, inputs: list, batch_size = 1) -> list:
        futures = []
        for i in range(0, len(inputs), batch_size):
            futures.append(self.__batch_predict_pool.submit(fn, np.stack(inputs[i : i + batch_size])))

        res = []
        for i, f in enumerate(futures):
            res += f.result()

        return res

    def __call__(self,
                 audio_path: Optional[str] = None, 
                 pcm: Optional[MusionPCM] = None,
                 save_cfg: Optional[SaveConfig] = None, 
                 overwrite: bool = False) -> dict:
        """
        core function for processing ONE audio file
        overwrite: whether to overwrite the existing result file, only works when save_cfg is provided.
        """
        if save_cfg and not save_cfg.keys:
            save_cfg.keys = self.result_keys

        if audio_path:
            logging.debug(f'Processing audio file: {audio_path} for task: {self.__class__.__name__}')
            if not overwrite and save_cfg and check_exist(audio_path, save_cfg):
                logging.info(f'File {audio_path} already processed, skip.')
                return {}

        start_time = time.time()
        res = self._process(audio_path, pcm)

        # Optionally save the result(s) to a file
        if save_cfg and audio_path:
            self.__save_res(res, save_cfg, audio_path)

        logging.debug(f"{self.__class__.__name__}__call__ execution time: {time.time() - start_time}")

        return res

    def __save_res(self, res: dict, save_cfg: SaveConfig, audio_path: str):
        if not os.path.exists(save_cfg.dir_path):
            os.makedirs(save_cfg.dir_path)
        audio_name = get_file_name(audio_path)

        for key in save_cfg.keys:
            if key not in self.result_keys:
                raise KeyError(f'Save key error! There is no {key} for task {self.__class__.__name__}.')
            if res[key] is None:
                logging.warning(f'Result for key: {key} of {audio_name} is None, will not save a file for it.')
                continue

            save_path = os.path.join(save_cfg.dir_path, audio_name + '.' + key)
            if '.wav' in key:
                torchaudio.save(save_path, res[key], self._feat_cfg.sample_rate, encoding="PCM_S", bits_per_sample=16)
            elif 'mid' in key:
                res[key].save(save_path)
            else:
                np.savetxt(save_path, res[key], fmt='%s')

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

    def _predict(self, model_input: Union[List[np.ndarray], np.ndarray]):
        if isinstance(model_input, np.ndarray):
            model_input = [model_input]
        ort_input = {i.name: d.astype(np.float32) for i, d in zip(self.__ort_session.get_inputs(), model_input)}
        return self.__ort_session.run(None, input_feed=ort_input)