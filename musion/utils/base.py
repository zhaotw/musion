from typing import Any, Optional, List, Union
import os
import dataclasses
import logging
import abc
import time

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import torchaudio
import torch
from tqdm import tqdm

from musion.utils.tools import *
from musion.utils.parallel import *

logging.basicConfig(level=logging.WARNING)

@dataclasses.dataclass
class MusionPCM:
    samples: torch.Tensor
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
    def __init__(self, task_class, **init_kwargs) -> None:
        self.__task_class = task_class
        self.__init_kwargs = init_kwargs
        self.__task = None
        
    @property
    def result_keys(self) -> List[str]:
        return self.get_task().result_keys

    def get_task(self):
        if self.__task is None:
            self.__task = self.__task_class(**self.__init_kwargs)
        return self.__task

    def __call__(self, path_input: Optional[Union[List[str], str]] = None, *args,
                 num_workers: int = 0, num_threads: int = 0, **kwargs: Any) -> dict:
        """
        num_workers: number of workers for parallel processing
        num_threads: number of threads for concurrent processing
        """

        if 'audio_path' in kwargs:  # for backward compatibility
            path_input = kwargs['audio_path']
            del kwargs['audio_path']

        if isinstance(path_input, List):
            res = self.__process_multi_file(path_input, num_workers, num_threads, **kwargs)
        elif isinstance(path_input, str) and os.path.isdir(path_input):
            res = self.__process_multi_file(get_file_list(path_input), num_workers, num_threads, **kwargs)
        else:
            res = self.get_task()(path_input, *args, **kwargs)

        return res

    def __process_multi_file(self, file_list: list, num_workers: int, num_threads: int, **call_kwargs) -> dict:
        res = {}

        if num_workers > 0:
            kwargs = {'init_kwargs': self.__init_kwargs, 'call_kwargs': call_kwargs}
            res = parallel_process(num_workers, self.__task_class, file_list, **kwargs)
        else:
            def serial_process(path_list, res):
                task = self.get_task()
                for path in tqdm(path_list, desc=f"Processing..."):
                    cur_res = task(path, **call_kwargs)
                    res[get_file_name(path)] = cur_res

            if num_threads == 0:
                serial_process(file_list, res)
            else:
                concurrent_process(num_threads, serial_process, file_list, res)

        return res

class TaskBase(metaclass=abc.ABCMeta):
    def __init__(self, device: Optional[str] = None) -> None:
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

    @abc.abstractmethod
    def _process(self, path_input: str, *args) -> dict:
        raise NotImplementedError("Subclass must implement this method.")

    @property
    @abc.abstractmethod
    def result_keys(self) -> List[str]:
        raise NotImplementedError("Subclass must implement this method.")

    @abc.abstractmethod
    def _save(self, key: str, save_path: str, res: dict) -> None:
        raise NotImplementedError("Subclass must implement this method.")

    def __call__(self,
                 path_input: Optional[str] = None,
                 *args,
                 save_cfg: Optional[SaveConfig] = None, 
                 overwrite: bool = False,
                 ) -> dict:
        """
        core function for processing ONE file
        overwrite: whether to overwrite the existing result file, only works when save_cfg is provided.
        """
        if save_cfg and not save_cfg.keys:
            save_cfg.keys = self.result_keys

        if path_input is not None:
            logging.debug(f'Processing file: {path_input} for task: {self.__class__.__name__}')
            if not overwrite and save_cfg and check_exist(path_input, save_cfg):
                logging.info(f'File {path_input} already processed, skip.')
                return {}

        start_time = time.time()
        res = self._process(path_input, *args)

        # Optionally save the result(s) to a file
        if save_cfg and path_input:
            self.__save_res(res, save_cfg, path_input)

        logging.debug(f"{self.__class__.__name__}__call__ execution time: {time.time() - start_time}")

        return res

    def __save_res(self, res: dict, save_cfg: SaveConfig, path_input: str):
        if not os.path.exists(save_cfg.dir_path):
            os.makedirs(save_cfg.dir_path, exist_ok=True)
        audio_name = get_file_name(path_input)

        for key in save_cfg.keys:
            if key not in self.result_keys:
                raise KeyError(f'Save key error! There is no {key} for task {self.__class__.__name__}.')
            if res[key] is None:
                logging.warning(f'Result for key: {key} of {audio_name} is None, will not save a file for it.')
                continue

            save_path = os.path.join(save_cfg.dir_path, audio_name + '.' + key)
            self._save(key, save_path, res)

class MusionBase(TaskBase):
    def __init__(self, device: Optional[str] = None) -> None:
        super().__init__(device)

    @property
    @abc.abstractmethod
    def _feat_cfg(self) -> FeatConfig:
        raise NotImplementedError("Subclass must implement this method.")

    @abc.abstractmethod
    def _process(self, audio_path: Optional[str] = None, pcm: Optional[MusionPCM] = None) -> dict:
        raise NotImplementedError("Subclass must implement this method.")

    def _load_pcm(self, audio_path: Optional[str] = None, pcm: Optional[MusionPCM] = None) -> MusionPCM:
        if audio_path is None and pcm is None:
            raise ValueError('Should provide either audio path or pcm to proceed.')
        if audio_path and pcm:
            logging.warning('Both audio path and pcm provided, will use audio path.')

        if audio_path:
            samples, sr = torchaudio.load(audio_path)
            pcm = MusionPCM(samples, sr)

        if pcm.sample_rate != self._feat_cfg.sample_rate:
            pcm.samples = torchaudio.transforms.Resample(pcm.sample_rate, self._feat_cfg.sample_rate)(
                pcm.samples)
            pcm.sample_rate = self._feat_cfg.sample_rate

        pcm.samples = convert_audio_channels(pcm.samples, 1 if self._feat_cfg.mono else 2)

        if pcm.samples.ndim == 1:
            pcm.samples = pcm.samples.unsqueeze(0)

        return pcm

    def __call__(self,
                 audio_path: Optional[str] = None, 
                 pcm: Optional[MusionPCM] = None,
                 save_cfg: Optional[SaveConfig] = None, 
                 overwrite: bool = False) -> dict:
        """
        core function for processing ONE audio file
        pcm: Pre-loaded PCM data, will be used if audio_path is None
        overwrite: whether to overwrite the existing result file, only works when save_cfg is provided.
        """
        return super().__call__(audio_path, pcm, save_cfg=save_cfg, overwrite=overwrite)

    def _save(self, key, save_path, res):
        np.savetxt(save_path, res[key], fmt='%s')
