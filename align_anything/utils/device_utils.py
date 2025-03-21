# Copyright 2025 PKU-Alignment Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""The device utility."""

import gc
import os
from typing import Tuple, Union

import torch
from transformers.utils import (
    is_torch_cuda_available,
    is_torch_mps_available,
    is_torch_npu_available,
    is_torch_xpu_available,
)


def get_current_device() -> 'torch.device':
    r"""
    Gets the current available device.
    """
    if is_torch_xpu_available():
        device = 'xpu:{}'.format(os.environ.get('LOCAL_RANK', '0'))
    elif is_torch_npu_available():
        device = 'npu:{}'.format(os.environ.get('LOCAL_RANK', '0'))
    elif is_torch_mps_available():
        device = 'mps:{}'.format(os.environ.get('LOCAL_RANK', '0'))
    elif is_torch_cuda_available():
        device = 'cuda:{}'.format(os.environ.get('LOCAL_RANK', '0'))
    else:
        device = 'cpu'

    return torch.device(device)


def set_device(device_id) -> str:
    r"""
    Sets the device.
    """
    if is_torch_xpu_available():
        device = f'xpu:{device_id}'
    elif is_torch_npu_available():
        device = f'npu:{device_id}'
    elif is_torch_mps_available():
        device = f'mps:{device_id}'
    elif is_torch_cuda_available():
        device = f'cuda:{device_id}'
    else:
        device = 'cpu'

    return device


def get_device_count() -> int:
    r"""
    Gets the number of available GPU or NPU devices.
    """
    if is_torch_xpu_available():
        return torch.xpu.device_count()
    elif is_torch_npu_available():
        return torch.npu.device_count()
    elif is_torch_cuda_available():
        return torch.cuda.device_count()
    else:
        return 0


def get_peak_memory() -> Tuple[int, int]:
    r"""
    Gets the peak memory usage for the current device (in Bytes).
    """
    if is_torch_npu_available():
        return torch.npu.max_memory_allocated(), torch.npu.max_memory_reserved()
    elif is_torch_cuda_available():
        return torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()
    else:
        return 0, 0


def is_gpu_or_npu_available() -> bool:
    r"""
    Checks if the GPU or NPU is available.
    """
    return is_torch_npu_available() or is_torch_cuda_available()


def torch_gc() -> None:
    r"""
    Frees up unused memory on supported hardware accelerators (GPU, NPU, MPS, etc.).

    This function performs two key steps:
    1. Invokes Python's garbage collector (`gc.collect()`) to reclaim memory from unreachable objects.
    2. Clears cached memory on the respective hardware device by calling the appropriate
       PyTorch API (`torch.xpu.empty_cache()`, `torch.npu.empty_cache()`, `torch.mps.empty_cache()`,
    """
    gc.collect()
    if is_torch_xpu_available():
        torch.xpu.empty_cache()
    elif is_torch_npu_available():
        torch.npu.empty_cache()
    elif is_torch_mps_available():
        torch.mps.empty_cache()
    elif is_torch_cuda_available():
        torch.cuda.empty_cache()


def torch_set_device(device: Union[torch.device, str, int, None]) -> None:
    r"""
    Sets the device for PyTorch.
    """
    if is_torch_npu_available():
        torch.npu.set_device(device)
    elif is_torch_cuda_available():
        torch.cuda.set_device(device)


def manual_seed_all(seed) -> None:
    r"""
    Sets the seed for generating random numbers.
    """
    torch.manual_seed(seed)
    if is_torch_cuda_available():
        torch.cuda.manual_seed_all(seed)
    if is_torch_npu_available():
        torch.npu.manual_seed_all(seed)
