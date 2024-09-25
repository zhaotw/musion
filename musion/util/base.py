from typing import Any, Optional, List, Union, Callable
import os
import dataclasses
import logging
import abc
import time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count

import numpy as np
import librosa
import torchaudio
import torch
import onnxruntime as ort
from tqdm import tqdm

from musion.util.tools import *

logging.basicConfig(level=logging.INFO)

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
    sample_rate: int
    n_fft: int
    hop_length: int
    f_min: Optional[float] = None
    f_max: Optional[float] = None
    n_mels: Optional[int] = None
    normalized: Union[str, bool] = False
    power: Optional[float] = 2.0
    norm: Optional[str] = None
    mel_scale: Optional[str] = 'htk'

class MusionBase(metaclass=abc.ABCMeta):
    def __init__(self, need_mono_pcm: bool, feat: torch.nn.Module, model_path: str) -> None:
        self.mono = need_mono_pcm

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file: {model_path} not found. Download it first following steps in README.")

        ort_options = ort.SessionOptions()
        ort_options.enable_cpu_mem_arena = False
        ort_options.enable_mem_pattern = False
        providers=[]
        if self.device == 'cuda':
            providers.append('CUDAExecutionProvider')
        providers.append('CPUExecutionProvider')
        self.__ort_session = ort.InferenceSession(model_path, sess_options=ort_options, providers=providers)

        num_threads = max(cpu_count() // 2, 1)
        self.pool = ThreadPoolExecutor(num_threads)

        self._feat = feat.to(self.device)

    def __load_pcm(self, audio_path: Optional[str] = None, pcm: Optional[MusionPCM] = None) -> MusionPCM:
        if audio_path is None and pcm is None:
            raise ValueError('Should provide either audio path or pcm to proceed.')

        if audio_path and pcm:
            logging.warning('Both audio path and pcm provided, will use audio path.')
        if audio_path:
            samples, sr = librosa.load(audio_path, sr=None, mono=self.mono)
            pcm = MusionPCM(samples, sr)

        if pcm.sample_rate != self._feat_cfg.sample_rate:
            pcm.samples = torchaudio.transforms.Resample(pcm.sample_rate, self._feat_cfg.sample_rate)(
                torch.from_numpy(pcm.samples))

        pcm.samples = np.asarray(pcm.samples)

        if pcm.samples.ndim == 1:
            pcm.samples = np.expand_dims(pcm.samples, 0)

        pcm.samples = convert_audio_channels(pcm.samples, 1 if self.mono else 2)

        return pcm

    def _predict(self, model_input: Union[List[np.ndarray], np.ndarray]):
        if isinstance(model_input, np.ndarray):
            model_input = [model_input]
        ort_input = {i.name: d.astype(np.float32) for i, d in zip(self.__ort_session.get_inputs(), model_input)}
        return self.__ort_session.run(None, input_feed=ort_input)

    def _batch_process(self, fn: Callable, inputs: list, batch_size = 1) -> list:
        futures = []
        for i in range(0, len(inputs), batch_size):
            futures.append(self.pool.submit(fn, np.stack(inputs[i : i + batch_size])))

        res = []
        for i, f in enumerate(tqdm(futures)):
            res += f.result()

        return res

    @property
    @abc.abstractmethod
    def _feat_cfg(self) -> FeatConfig:
        raise NotImplementedError("Subclass must implement this method.")

    @abc.abstractmethod
    def _process(self, samples: np.ndarray) -> dict:
        raise NotImplementedError("Subclass must implement this method.")

    def __call__(self, *, audio_path: Optional[Union[List[str], str]] = None, pcm: Optional[MusionPCM] = None,
                 save_cfg: Optional[SaveConfig] = None, num_threads: int = 0, **kwargs: Any) -> dict:
        if isinstance(audio_path, List):
            res = self.__process_multi_file(audio_path, num_threads, save_cfg)
        elif isinstance(audio_path, str) and os.path.isdir(audio_path):
            res = self.__process_multi_file(get_file_list(audio_path), num_threads, save_cfg)
        else:
            res = self.__process_single_file(audio_path, pcm, save_cfg)

        return res

    def __process_single_file(self, audio_path: Optional[str] = None, pcm: Optional[MusionPCM] = None,
                 save_cfg: Optional[SaveConfig] = None) -> dict:
        if audio_path:
            logging.info(f'Processing audio file: {audio_path} for task: {self.__class__.__name__}')

        start_time = time.time()

        pcm = self.__load_pcm(audio_path, pcm)
        res = self._process(pcm.samples)

        # Optionally save the result(s) to a file
        if save_cfg and audio_path:
            self.__save_res(res, save_cfg, audio_path)

        logging.debug(f"__call__ execution time: {time.time() - start_time}")

        return res

    def __process_multi_file(self, file_list: list, num_threads: int, save_cfg: Optional[SaveConfig] = None) -> dict:
        res = {}
        audio_durations = []
        start_time = time.time()
        res = {}

        def serial_process(file_list, audio_durations, res):
            for audio_path in file_list:
                audio_durations.append(librosa.get_duration(filename=audio_path))
                cur_res = self.__process_single_file(audio_path=audio_path, save_cfg=save_cfg)
                res[get_file_name(audio_path)] = cur_res

        if num_threads == 0:
            serial_process(file_list, audio_durations, res)
        else:
            parallel_process(num_threads, serial_process, file_list, audio_durations, res)

        process_time = time.time() - start_time
        total_audio_duration = sum(audio_durations)
        logging.info(f'Done! RTF: {process_time / total_audio_duration}')

        return res

    def __save_res(self, res: dict, save_cfg: SaveConfig, audio_path: str):
        if not os.path.exists(save_cfg.dir_path):
            os.makedirs(save_cfg.dir_path)
        audio_name = get_file_name(audio_path)
        save_cfg.keys = save_cfg.keys if save_cfg.keys else self.result_keys
        for key in save_cfg.keys:
            if key not in self.result_keys:
                raise KeyError(f'Save key error! There is no {key} for task {self.__class__.__name__}.')
            if res[key] is None:
                logging.warning(f'Result for key: {key} is None, will not save a file for it.')
                continue

            save_path = os.path.join(save_cfg.dir_path, audio_name + '.' + key)
            if '.wav' in key:
                torchaudio.save(save_path, res[key], self._feat_cfg.sample_rate, encoding="PCM_S", bits_per_sample=16)
            elif 'mid' in key:
                res[key].save(save_path)
            else:
                np.savetxt(save_path, res[key], fmt='%s')

    @property
    @abc.abstractmethod
    def result_keys(self) -> List[str]:
        raise NotImplementedError("Subclass must implement this method.")
