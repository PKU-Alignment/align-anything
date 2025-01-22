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


import math
import os
from typing import Any

import deepspeed
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
        if zero_stage == 2:
            raise ValueError('MiniCPM-O does not support ZeRO stage 2')
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH, trust_remote_code=True)
        self.system_prompt = 'You are a helpful assistant. You can accept video, audio and text input and output voice and text. '
        os.environ['MULTI_IMAGES_INFERENCE_MODELS'] = 'Yes'
        deepspeed.zero.register_external_parameter(self, self.apm.embed_positions.weight)

    @staticmethod
    def model_additional_kwargs(modality: list[str]):
        return {
            'init_audio': 'audio' in modality,
            'init_tts': False,
            'init_vision': True,
            'vision_batch_size': 256,
        }

    def apply_chat_template(
        self, messages: list[dict[str, Any]], add_generation_prompt: bool = False
    ) -> dict[str, Any]:
        prompt_list = []
        system_prompt = {'role': 'system', 'content': self.system_prompt}
        prompt_list.append(system_prompt)
        for message in messages:
            role = message['role']
            contents = message['content']
            for idx, content in enumerate(contents):
                if content['type'] == 'text':
                    msg = {'role': role, 'content': content['text']}
                    if role == 'user':
                        if idx - 1 >= 0:
                            if contents[idx - 1]['type'] == 'image':
                                msg['content'] += '(<image>./</image>)'
                            elif contents[idx - 1]['type'] == 'audio':
                                msg['content'] += '(<audio>./</audio>)'
                        elif idx + 1 < len(contents):
                            if contents[idx + 1]['type'] == 'image':
                                msg['content'] += '(<image>./</image>)'
                            elif contents[idx + 1]['type'] == 'audio':
                                msg['content'] += '(<audio>./</audio>)'
                    prompt_list.append(msg)

        return self.tokenizer.apply_chat_template(
            prompt_list,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            chat_template=self.default_tts_chat_template if self.config.init_audio else None,
        )

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        pixel_values: torch.Tensor | list = [],
        tgt_sizes: torch.Tensor | None = None,
        audio_features: torch.Tensor | None = None,
        audio_feature_lens: torch.Tensor | None = None,
        image_bound: torch.Tensor | None = None,
        audio_bounds: torch.Tensor | None = None,
        spk_bounds: torch.Tensor | None = None,
        vision_hidden_states: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **kwargs,
    ):
        batch_size = input_ids.shape[0]
        model_inputs = {
            'input_ids': input_ids,
            'audio_features': audio_features,
            'audio_feature_lens': audio_feature_lens,
            'image_bound': image_bound or [[]] * batch_size,
            'audio_bounds': audio_bounds,
            'spk_bounds': spk_bounds,
        }
        dtype = self.llm.model.embed_tokens.weight.dtype
        device = self.llm.model.embed_tokens.weight.device
        if len(pixel_values) == 0:
            vision_hidden_states = []
            dummy_image = torch.zeros((batch_size, 3, 224, 224), device=device, dtype=dtype)
            tgt_sizes = torch.Tensor(
                [[(224 // self.config.patch_size), math.ceil(224 / self.config.patch_size)]]
            ).type(torch.int32)
            dummy_feature = self.resampler(self.vpm(dummy_image).last_hidden_state, tgt_sizes)
            for _ in range(batch_size):
                vision_hidden_states.append(dummy_feature)
        if vision_hidden_states is None:
            model_inputs['pixel_values'] = pixel_values
            model_inputs['tgt_sizes'] = tgt_sizes
        else:
            model_inputs['vision_hidden_states'] = vision_hidden_states

        with torch.no_grad():
            position_ids = torch.arange(input_ids.size(1)).long().repeat(batch_size, 1)
            model_inputs['position_ids'] = position_ids.to(device)

        return super().forward(
            data=model_inputs,
            attention_mask=attention_mask,
            labels=labels.to(torch.long) if labels is not None else None,
            **kwargs,
        )
