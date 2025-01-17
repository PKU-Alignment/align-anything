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

import torch.utils.checkpoint
from transformers import AutoConfig, AutoTokenizer
from transformers.dynamic_module_utils import get_class_from_dynamic_module


MODEL_NAME_OR_PATH = os.environ.get('MODEL_NAME_OR_PATH', 'openbmb/MiniCPM-o-2_6')
CONFIG = AutoConfig.from_pretrained(MODEL_NAME_OR_PATH, trust_remote_code=True)
CLASS_REF = CONFIG.auto_map['AutoModel']
MiniCPMO = get_class_from_dynamic_module(CLASS_REF, MODEL_NAME_OR_PATH)

class AccustomedMiniCPMO(MiniCPMO):

    def __init__(self, config: AutoConfig):
        super().__init__(config)
        zero_stage = int(os.environ.get('ZERO_STAGE', '0'))
        if zero_stage == 3:
            raise ValueError('MiniCPM-V does not support ZeRO stage 3')
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME_OR_PATH, trust_remote_code=True
        )
