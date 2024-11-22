# Copyright 2024 PKU-Alignment Team. All Rights Reserved.
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


import importlib

from typing import Any

from collections import OrderedDict

from transformers.models.auto.auto_factory import (
    _BaseAutoModelClass,
    _LazyAutoMapping,
    getattribute_from_module,
)

from transformers.models.auto.configuration_auto import (
    CONFIG_MAPPING_NAMES,
    model_type_to_module_name,
)

from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES

class _LazyAutoMappingInAlignAnything(_LazyAutoMapping):
    def _load_attr_from_module(self, model_type: str, attr: str) -> Any:
        module_name = model_type_to_module_name(model_type)
        if module_name not in self._modules:
            try:
                self._modules[module_name] = importlib.import_module(
                    f'.{module_name}',
                    'align_anything.models',
                )
            except ImportError:
                self._modules[module_name] = importlib.import_module(f".{module_name}", "transformers.models")
        return getattribute_from_module(self._modules[module_name], attr)

MODEL_FOR_SCORE_MAPPING_NAMES: OrderedDict[str, str] = OrderedDict(
    [
        # Score model mapping
        ('llama', 'AccustomedLlamaRewardModel'),
        ('llava', 'AccustomedLlavaRewardModel'),
        ('qwen2_audio', 'AccustomedQwen2AudioRewardModel'),
        ('chameleon', 'AccustomedChameleonRewardModel'),
        ('qwen2_vl', 'AccustomedQwen2VLRewardModel'),
    ],
)

MODEL_MAPPING_NAMES: OrderedDict[str, str] = OrderedDict(
    [
        # Score model mapping
        ('llama', 'AccustomedLlamaModel'),
        ('llava', 'AccustomedLlavaModel'),
        ('qwen2_audio', 'AccustomedQwen2AudioModel'),
        ('chameleon', 'AccustomedChameleonModel'),
        ('qwen2_vl', 'AccustomedQwen2VLModel'),
        ('modeling_emu3.mllm.modeling_emu3', 'Emu3ForCausalLM'),
    ],
)

MODEL_FOR_SCORE_MAPPING: OrderedDict[str, Any] = _LazyAutoMappingInAlignAnything(
    CONFIG_MAPPING_NAMES,
    MODEL_FOR_SCORE_MAPPING_NAMES,
)

MODEL_MAPPING: OrderedDict[str, Any] = _LazyAutoMappingInAlignAnything(
    CONFIG_MAPPING_NAMES,
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES | MODEL_MAPPING_NAMES,
)

class AnyModelForScore(_BaseAutoModelClass):
    _model_mapping: OrderedDict[str, Any] = MODEL_FOR_SCORE_MAPPING

class AnyModel(_BaseAutoModelClass):
    _model_mapping: OrderedDict[str, Any] = MODEL_MAPPING
