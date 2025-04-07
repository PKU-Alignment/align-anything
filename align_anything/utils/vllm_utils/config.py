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
"""Configuration for vLLM integration."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class VLLMConfig:
    """Configuration for vLLM integration."""

    use_vllm: bool = False
    """Whether to use vLLM for accelerated sampling."""

    vllm_num_engines: int = 1
    """Number of vLLM engines to create."""

    vllm_tensor_parallel_size: int = 1
    """Tensor parallel size for vLLM."""

    vllm_enable_prefix_caching: bool = True
    """Whether to enable prefix caching in vLLM."""

    vllm_enforce_eager: bool = True
    """Whether to enforce eager execution in vLLM."""

    vllm_max_model_len: Optional[int] = None
    """Maximum model length for vLLM. If None, use model_max_length."""

    vllm_gpu_memory_utilization: float = 0.9
    """GPU memory utilization for vLLM."""

    vllm_enable_sleep: bool = False
    """Whether to enable sleep mode in vLLM."""
