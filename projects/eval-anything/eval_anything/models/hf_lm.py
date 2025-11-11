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

"""
(t2t)支持transformers+accelerate推理
"""

import gc
from typing import Any, Dict, List

import torch
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval_anything.models.base_model import BaseModel
from eval_anything.utils.data_type import InferenceInput, InferenceOutput
from eval_anything.utils.register import TemplateRegistry as get_template
from eval_anything.utils.utils import get_messages


class AccelerateModel(BaseModel):
    def __init__(self, model_cfgs: Dict[str, Any], infer_cfgs, **kwargs):
        self.model_cfgs = model_cfgs
        self.infer_cfgs = infer_cfgs

        self.model_id = self.model_cfgs.model_id
        self.model_name_or_path = self.model_cfgs.model_name_or_path
        self.chat_template = self.model_cfgs.chat_template

        self.model_max_length = self.infer_cfgs.model_max_length
        self.max_new_tokens = self.infer_cfgs.max_new_tokens

        self.task2details = {}
        self.detailed_filename = f'{self.model_id}_detailed'
        self.brief_filename = f'{self.model_id}_brief'

        self.init_model()

    def init_model(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name_or_path)

        self.accelerator = Accelerator()
        self.model = self.accelerator.prepare(self.model)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def generation(
        self, inputs: Dict[str, List[InferenceInput]]
    ) -> Dict[str, List[InferenceOutput]]:
        return self._generation(inputs)

    def _generation(self, input_list: List[InferenceInput]) -> Dict[str, List[InferenceOutput]]:
        if self.chat_template:
            self.template = get_template(self.chat_template)
            prompts = [
                self.template.system_prompt
                + self.template.user_prompt.format(input=input.text)
                + self.template.assistant_prompt.format(output='')
                for input in input_list
            ]
            encoded_inputs = self.tokenizer(
                prompts,
                padding=True,
                truncation=True,
                max_length=self.model_max_length,
                return_tensors='pt',
            )

            outputs = self.model.generate(
                input_ids=encoded_inputs['input_ids'].to(self.accelerator.device),
                attention_mask=encoded_inputs['attention_mask'].to(self.accelerator.device),
                max_length=self.model_max_length,
                num_return_sequences=1,
            )
            output_ids = [output[encoded_inputs['input_ids'].shape[-1] :] for output in outputs]
            responses = [
                self.tokenizer.decode(output_id, skip_special_tokens=True)
                for output_id in output_ids
            ]

        else:
            self.modality = 't2t'
            prompts = [get_messages(self.modality, input) for input in input_list]
            input_ids = self.tokenizer.apply_chat_template(
                prompts, padding=True, add_generation_prompt=True, return_tensors='pt'
            )

            outputs = self.model.generate(
                input_ids=input_ids.to(self.accelerator.device),
                max_length=self.model_max_length,
                num_return_sequences=1,
            )
            output_ids = [output[input_ids.shape[-1] :] for output in outputs]
            responses = [
                self.tokenizer.decode(output_id, skip_special_tokens=True)
                for output_id in output_ids
            ]

        inference_outputs = [
            InferenceOutput.from_hf_output(
                task=input.task,
                uuid=input.uuid,
                response=response,
                hf_output=output_id,
                store_raw=True,
            )
            for input, response, output_id in zip(input_list, responses, output_ids)
        ]

        return inference_outputs

    def shutdown_model(self):
        del self.model
        self.model = None
        del self.accelerator
        self.accelerator = None

        gc.collect()
        torch.cuda.empty_cache()
