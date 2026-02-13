from typing import Optional, List, Union, Callable
import os
import ctypes
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from multiprocessing import cpu_count
import logging

import onnxruntime as ort
import numpy as np
import torch

from musion.utils.base import MusionBase, MusionPCM

# CUDA memory copy for GPU-to-GPU transfer
_cuda_lib = None
def _get_cuda_memcpy():
    global _cuda_lib
    if _cuda_lib is None:
        _cuda_lib = ctypes.CDLL('libcudart.so')
        _cuda_lib.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
        _cuda_lib.cudaMemcpy.restype = ctypes.c_int
    return _cuda_lib.cudaMemcpy

_CUDA_MEMCPY_D2D = 3  # cudaMemcpyDeviceToDevice

_TRT_LIBPATH_SET = False
_TRT_AVAILABLE = None

def _ensure_trt_library_path() -> bool:
    global _TRT_LIBPATH_SET
    if _TRT_LIBPATH_SET:
        return True
    for base in tuple(Path(p) for p in os.sys.path if p):
        candidate = base / "tensorrt_libs"
        if candidate.is_dir():
            ld_path = os.environ.get("LD_LIBRARY_PATH", "")
            if str(candidate) not in ld_path.split(":"):
                os.environ["LD_LIBRARY_PATH"] = f"{candidate}:{ld_path}" if ld_path else str(candidate)
            _TRT_LIBPATH_SET = True
            return True
    return False

def _trt_available() -> bool:
    global _TRT_AVAILABLE
    if _TRT_AVAILABLE is not None:
        return _TRT_AVAILABLE
    if "TensorrtExecutionProvider" not in ort.get_available_providers():
        _TRT_AVAILABLE = False
        return _TRT_AVAILABLE
    _ensure_trt_library_path()
    try:
        ctypes.CDLL("libnvinfer.so.10")
        _TRT_AVAILABLE = True
    except OSError:
        _TRT_AVAILABLE = False
    return _TRT_AVAILABLE

class OrtMusionBase(MusionBase):
    def __init__(self, model_path: str, device: Optional[str] = None, trt_fp16_enable=False) -> None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file: {model_path} not found. Download it first following steps in README.")

        super().__init__(device)

        ort_options = ort.SessionOptions()
        ort_options.enable_cpu_mem_arena = True
        ort_options.enable_mem_pattern = True
        ort_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        providers = []
        if 'cuda' in self.device:
            self.device_id = 0
            if 'cuda:' in self.device:
                self.device_id = int(self.device.split(':')[1])

            if _trt_available():
                trt_options = {
                    "device_id": self.device_id,
                    "trt_fp16_enable": trt_fp16_enable,
                    "trt_engine_cache_enable": True,
                    "trt_engine_cache_path": './trt_cache'
                }
                providers.append(("TensorrtExecutionProvider", trt_options))
            else:
                cuda_options = {
                    "device_id": self.device_id,
                    "use_tf32": True,
                }
                providers.append(('CUDAExecutionProvider', cuda_options))

        if not providers:
            providers.append('CPUExecutionProvider')
        logging.info(f"Using providers: {providers}")
        self.__ort_session = ort.InferenceSession(model_path, sess_options=ort_options, providers=providers)

        num_threads = max(cpu_count() // 2, 1)
        self.__batch_predict_pool = ThreadPoolExecutor(num_threads)

    def _load_pcm(self, audio_path: Optional[str] = None, pcm: Optional[MusionPCM] = None) -> MusionPCM:
        pcm = super()._load_pcm(audio_path, pcm)
        pcm.samples = pcm.samples.numpy()
        return pcm

    def _batch_process(self, fn: Callable, inputs: list, batch_size = 1) -> list:
        """
        Remember to remove batch dim for inputs before calling this function
        """
        futures = []
        for i in range(0, len(inputs), batch_size):
            if isinstance(inputs[0], np.ndarray): 
                input = np.stack(inputs[i : i + batch_size])
            else:
                input = torch.stack(inputs[i : i + batch_size])
            futures.append(self.__batch_predict_pool.submit(fn, input))

        res = []
        for f in futures:
            res.append(f.result())
        return res

    def _predict(self, model_input: Union[List[np.ndarray], np.ndarray]):
        if isinstance(model_input, np.ndarray):
            model_input = [model_input]
        ort_input = {}
        for i, d in zip(self.__ort_session.get_inputs(), model_input):
            ort_input[i.name] = d.astype(np.float32)
        return self.__ort_session.run(None, input_feed=ort_input)

    def _predict_torch(self, model_input):
        if torch.is_tensor(model_input):
            model_input = [model_input]

        io_binding = self.__ort_session.io_binding()
        input_ortvalues = []  # prevent memory from being freed by GC
        
        for i, t in zip(self.__ort_session.get_inputs(), model_input):
            if t.dtype != torch.float32:
                t = t.float()
            if not t.is_contiguous():
                t = t.contiguous()

            # 使用 bind_input 直接绑定 tensor 的内存
            io_binding.bind_input(
                name=i.name,
                device_type='cuda',
                device_id=self.device_id,
                element_type=np.float32,
                shape=tuple(t.shape),
                buffer_ptr=t.data_ptr(),
            )
            input_ortvalues.append(t)  # keep reference to prevent memory from being freed

        for o in self.__ort_session.get_outputs():
            io_binding.bind_output(o.name, 'cuda', self.device_id)

        # TODO: fix
        # Key fix 1: Use ONNX Runtime's official sync method to guarantee input data is fully ready
        # io_binding.synchronize_inputs()
        # Additional fix: Manually accessing input data to ensure TensorRT is also synchronized (TensorRT may require extra sync)
        for t in input_ortvalues:
            _ = t.sum().item()

        self.__ort_session.run_with_iobinding(io_binding)

        io_binding.synchronize_outputs()

        ort_outputs = io_binding.get_outputs()
        
        results = []
        cudaMemcpy = _get_cuda_memcpy()
        for o in ort_outputs:
            # OrtValue does not support to_dlpack
            # Use cudaMemcpy for GPU-to-GPU copying to avoid CPU transfer
            shape = tuple(o.shape())
            output_tensor = torch.empty(shape, dtype=torch.float32, device=f'cuda:{self.device_id}')
            nbytes = output_tensor.numel() * 4  # float32 = 4 bytes
            cudaMemcpy(output_tensor.data_ptr(), o.data_ptr(), nbytes, _CUDA_MEMCPY_D2D)
            results.append(output_tensor)

        return results