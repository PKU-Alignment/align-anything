# Copyright 2024 PKU-Alignment Team. All Rights Reserved.
#
# This code is inspired by the HuggingFace's Transformers library.
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llava/modeling_llava.py
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

import os
from typing import Any

from transformers import AutoConfig, AutoTokenizer
from transformers.dynamic_module_utils import get_class_from_dynamic_module


MODEL_NAME_OR_PATH = os.environ.get('MODEL_NAME_OR_PATH', 'baichuan-inc/Baichuan-M1-14B-Instruct')
CONFIG = AutoConfig.from_pretrained(MODEL_NAME_OR_PATH, trust_remote_code=True)
CLASS_REF = CONFIG.auto_map['AutoModelForCausalLM']
BaichuanM1 = get_class_from_dynamic_module(CLASS_REF, MODEL_NAME_OR_PATH)


class AccustomedBaichuanM1(BaichuanM1):

    def __init__(self, config: AutoConfig):
        super().__init__(config)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH, trust_remote_code=True)
        self.system_prompt = ''

    def apply_chat_template(
        self, messages: list[dict[str, Any]], add_generation_prompt: bool = False
    ) -> dict[str, Any]:
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt
        )
