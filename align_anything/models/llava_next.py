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


from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from transformers import LlavaNextPreTrainedModel
from transformers.models.llava_next.modeling_llava_next import (
    LlavaNextCausalLMOutputWithPast,
    LlavaNextForConditionalGeneration,
    image_size_to_num_patches,
)


class AccustomedLlavaNextModel(LlavaNextForConditionalGeneration):

    @property
    def infer_required_keys(self) -> list[str]:
        return ['input_ids', 'attention_mask', 'pixel_values', 'labels', 'image_sizes']
    
    @property
    def processor_available(self):
        return True

    def infer_batch(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Return the dict used for model inference"""
        return {
            'input_ids': batch['input_ids'],
            'attention_mask': batch['attention_mask'],
            'pixel_values': batch['pixel_values'],
            'labels': batch.get('labels'),
            'image_sizes': batch.get('image_sizes'),
        }
