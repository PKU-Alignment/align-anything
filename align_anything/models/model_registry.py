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
from collections import OrderedDict
from typing import Any

from transformers import AutoConfig
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
                self._modules[module_name] = importlib.import_module(
                    f'.{module_name}', 'transformers.models'
                )
        return getattribute_from_module(self._modules[module_name], attr)


def get_model_class_for_trust_remote_code(model_type, model_mapping_names):
    model_class_name = model_mapping_names[model_type]
    try:
        model_class = getattr(
            importlib.import_module(f'.{model_type}', 'align_anything.models'), model_class_name
        )
    except ImportError:
        model_class = getattr(
            importlib.import_module(f'.{model_type}', 'transformers.models'), model_class_name
        )
    return model_class


MODEL_FOR_SCORE_MAPPING_NAMES: OrderedDict[str, str] = OrderedDict(
    [
        # Score model mapping
        ('llama', 'AccustomedLlamaRewardModel'),
        ('mllama', 'AccustomedMllamaRewardModel'),
        ('llava', 'AccustomedLlavaRewardModel'),
        ('llava_next', 'AccustomedLlavaNextRewardModel'),
        ('qwen2_audio', 'AccustomedQwen2AudioRewardModel'),
        ('chameleon', 'AccustomedChameleonRewardModel'),
        ('qwen2_vl', 'AccustomedQwen2VLRewardModel'),
        ('idefics2', 'AccustomedIdefics2RewardModel'),
    ],
)

MODEL_MAPPING_NAMES: OrderedDict[str, str] = OrderedDict(
    [
        # Score model mapping
        ('llama', 'AccustomedLlamaModel'),
        ('mllama', 'AccustomedMllamaModel'),
        ('llava', 'AccustomedLlavaModel'),
        ('llava_next', 'AccustomedLlavaNextModel'),
        ('qwen2_audio', 'AccustomedQwen2AudioModel'),
        ('chameleon', 'AccustomedChameleonModel'),
        ('qwen2_vl', 'AccustomedQwen2VLModel'),
        ('modeling_emu3.mllm.modeling_emu3', 'Emu3ForCausalLM'),
        ('llava_next_video', 'AccustomedLlavaNextVideoModel'),
        ('idefics2', 'AccustomedIdefics2Model'),
    ],
)

TRUST_REMOTE_CODE_MODEL_MAPPING_NAMES = OrderedDict(
    [
        ('minicpmv', 'AccustomedMiniCPMV'),
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

TRUST_REMOTE_CODE_MODEL_MAPPING: OrderedDict[str, Any] = _LazyAutoMappingInAlignAnything(
    CONFIG_MAPPING_NAMES,
    TRUST_REMOTE_CODE_MODEL_MAPPING_NAMES,
)


class AnyModelForScore(_BaseAutoModelClass):
    _model_mapping: OrderedDict[str, Any] = MODEL_FOR_SCORE_MAPPING


class AnyModel(_BaseAutoModelClass):
    _model_mapping: OrderedDict[str, Any] = MODEL_MAPPING

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        *model_args,
        trust_remote_code=False,
        code_revision=None,
        commit_hash=None,
        **kwargs,
    ):
        config, kwargs = AutoConfig.from_pretrained(
            pretrained_model_name_or_path,
            return_unused_kwargs=True,
            trust_remote_code=trust_remote_code,
            code_revision=code_revision,
            _commit_hash=commit_hash,
            **kwargs,
        )
        model_type = config.model_type
        if model_type in TRUST_REMOTE_CODE_MODEL_MAPPING_NAMES:
            return get_model_class_for_trust_remote_code(
                model_type, TRUST_REMOTE_CODE_MODEL_MAPPING_NAMES
            ).from_pretrained(
                pretrained_model_name_or_path,
                *model_args,
                trust_remote_code=trust_remote_code,
                **kwargs,
            )
        return super().from_pretrained(
            pretrained_model_name_or_path,
            *model_args,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )
