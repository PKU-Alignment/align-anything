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

from typing import Any, Dict, List

from eval_anything.utils.data_type import InferenceInput, InferenceOutput


MODEL_MAP = {
    'vllm_LM': 'vllm_lm',
    'vllm_MM': 'vllm_mm',
    'hf_LM': 'hf_lm',
    'hf_MM': 'hf_mm',
    'hf_VLA': 'hf_vla',
    'api_LM': 'api_lm',
    'api_MM': 'api_mm',
}
CLASS_MAP = {
    'vllm_LM': 'vllmLM',
    'vllm_MM': 'vllmMM',
    'hf_LM': 'HFLM',
    'hf_MM': 'HFMM',
    'hf_VLA': 'HFVLA',
    'api_LM': 'APILM',
    'api_MM': 'APIMM',
}


class BaseModel:
    def __init__(self, model_cfgs: Dict[str, Any]):
        self.model_cfgs = model_cfgs
        self.init_model()

    def init_model(self):
        pass

    def generation(
        self, inputs: Dict[str, List[InferenceInput]]
    ) -> Dict[str, List[InferenceOutput]]:
        pass

    def _generation(
        self, inputs: Dict[str, List[InferenceInput]]
    ) -> Dict[str, List[InferenceOutput]]:
        pass

    def shutdown_model(self):
        pass
