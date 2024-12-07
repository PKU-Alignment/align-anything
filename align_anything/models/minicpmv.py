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
import torch
import torch.utils.checkpoint
from typing import Any
from PIL import Image

from transformers import AutoConfig, AutoTokenizer
from transformers.dynamic_module_utils import get_class_from_dynamic_module

MODEL_NAME_OR_PATH = os.environ.get("MODEL_NAME_OR_PATH", "openbmb/MiniCPM-V")
CONFIG = AutoConfig.from_pretrained(MODEL_NAME_OR_PATH, trust_remote_code=True)
CLASS_REF = CONFIG.auto_map["AutoModelForCausalLM"]
MiniCPMV = get_class_from_dynamic_module(CLASS_REF, MODEL_NAME_OR_PATH)

class AccustomedMiniCPMV(MiniCPMV):

    def __init__(self, config: AutoConfig):
        super().__init__(config)
        zero_stage = int(os.environ['ZERO_STAGE'])
        if zero_stage == 3:
            raise ValueError("MiniCPM-V does not support ZeRO stage 3")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH, trust_remote_code=True)

    @property
    def infer_required_keys(self) -> list[str]:
        return ['input_ids', 'attention_mask', 'images', 'labels']
    
    @property
    def processor_available(self):
        return False

    def infer_batch(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Return the dict used for model inference"""
        return {
            'input_ids': batch['input_ids'],
            'attention_mask': batch['attention_mask'],
            'images': batch['images'],
            'labels': batch.get('labels'),
        }

    def get_input_embeddings(self):
        return self.llm.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.llm.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.llm.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.llm.set_output_embeddings(new_embeddings)

    def apply_chat_template(self, messages: list[dict[str, Any]], add_generation_prompt: bool = False) -> dict[str, Any]:
        conversation = ''
        for message in messages:
            role = message['role']
            contents = message['content']
            for content in contents:
                if content['type'] == 'text':   
                    if role == 'user':
                        content = '<用户>' + self.tokenizer.im_start + self.tokenizer.unk_token * self.config.query_num + self.tokenizer.im_end + '\n' + content['text']
                    else:
                        content = '<AI>' + '\n' + content['text']
                    conversation += content
        if add_generation_prompt:
            conversation += '<AI>'
        return conversation

    def forward(
            self,
            input_ids: torch.LongTensor | None = None,
            attention_mask: torch.Tensor | None = None,
            images: list[Image.Image] | None = None,
            labels: torch.Tensor | None = None,
            **kwargs,
        ):
        image_bound = []
        for input_id in input_ids:
            image_start_tokens = torch.where(input_id == self.tokenizer.im_start_id)[0] + 1
            image_end_tokens = torch.where(input_id == self.tokenizer.im_end_id)[0]
            valid_image_nums = max(len(image_start_tokens), len(image_end_tokens))
            image_bound.append(torch.hstack(
                [image_start_tokens[:valid_image_nums].unsqueeze(-1),
                image_end_tokens[:valid_image_nums].unsqueeze(-1)]
            ))

        batch_size = input_ids.size(0)
        pixel_values_list = []
        for i in range(batch_size): 
            pixel_values = [self.transform(images[i].convert('RGB'))]
            pixel_values_list.append(torch.stack(pixel_values).to(self.device))
        
        data = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'pixel_values': pixel_values_list,
            'labels': labels,
            'image_bound': image_bound,
        }
        vllm_embedding, _ = self.get_vllm_embedding(data)

        return self.llm(
            input_ids=None,
            inputs_embeds=vllm_embedding,
            **kwargs
        )

