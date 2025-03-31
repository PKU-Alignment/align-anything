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
"""vLLM utilities for accelerated sampling."""

from align_anything.utils.vllm_utils.vllm_engine import batch_vllm_engine_call, create_vllm_engines
from align_anything.utils.vllm_utils.vllm_sampling import generate_with_vllm


__all__ = ['create_vllm_engines', 'batch_vllm_engine_call', 'generate_with_vllm']
