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
(multi-modal)支持transformers+accelerate推理
"""

import gc
from typing import Any, Dict, List

import torch
from accelerate import Accelerator
from PIL import Image
from transformers import AutoConfig, AutoProcessor

from eval_anything.models.base_model import BaseModel
from eval_anything.utils.data_type import InferenceInput, InferenceOutput
from eval_anything.utils.register import TemplateRegistry as get_template
from eval_anything.utils.utils import get_messages


class AccelerateMultimodalModel(BaseModel):
    def __init__(self, model_cfgs: Dict[str, Any], infer_cfgs, **kwargs):
        self.model_cfgs = model_cfgs
        self.infer_cfgs = infer_cfgs

        self.model_id = self.model_cfgs.model_id
        self.model_name_or_path = self.model_cfgs.model_name_or_path
        self.chat_template = self.model_cfgs.chat_template
        self.template = get_template(self.chat_template)

        self.model_max_length = self.infer_cfgs.model_max_length
        self.max_new_tokens = self.infer_cfgs.max_new_tokens

        self.task2details = {}
        self.detailed_filename = f'{self.model_id}_detailed'
        self.brief_filename = f'{self.model_id}_brief'

        self.init_model()

    def init_model(self) -> None:
        self.model_config = AutoConfig.from_pretrained(self.model_name_or_path)
        self.ModelForConditionalGeneration = self.model_config.architectures[0]
        self.model_class = globals().get(self.ModelForConditionalGeneration)

        if self.model_class:
            self.model = getattr(self.model_class, 'from_pretrained')(self.model_name_or_path)
        else:
            raise ValueError(f'Model class {self.ModelForConditionalGeneration} not found.')

        self.processor = AutoProcessor.from_pretrained(self.model_name_or_path)

        self.accelerator = Accelerator()
        self.model = self.accelerator.prepare(self.model)

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
        else:
            prompts = [
                self.processor.apply_chat_template(
                    get_messages(self.modality, input.text), add_generation_prompt=True
                )
                for input in input_list
            ]

        if input_list and input_list[0].mm_data and input_list[0].mm_data[0].url:
            image_files = [input.mm_data[0].url for input in input_list]
            self.modality = input_list[0].mm_data[0].modality
        elif input_list and input_list[0].mm_data and input_list[0].mm_data[0].file:
            image_files = [input.mm_data[0].file for input in input_list]
            self.modality = input_list[0].mm_data[0].modality
        else:
            raise ValueError("Each input item must have either 'url' or 'file'.")

        outputs = []
        responses = []
        for prompt, image_file in zip(prompts, image_files):
            if isinstance(image_file, Image.Image):
                image = image_file
            elif isinstance(image_file, str):
                image = Image.open(image_file).convert('RGB')
            else:
                raise ValueError('image_file is neither a PIL Image nor a string.')

            inputs = self.processor(images=image, text=prompt, return_tensors='pt')
            inputs = inputs.to(self.accelerator.device)
            output_id = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
            outputs.append(output_id)
            output_id_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, output_id)
            ]
            response = self.processor.batch_decode(
                output_id_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            responses.append(response[0])

        inference_outputs = [
            InferenceOutput.from_hf_output(
                task=input.task,
                uuid=input.uuid,
                response=response,
                hf_output=output,
                store_raw=True,
            )
            for input, response, output in zip(input_list, responses, outputs)
        ]

        return inference_outputs

    def shutdown_model(self):
        del self.model
        self.model = None
        del self.accelerator
        self.accelerator = None

        gc.collect()
        torch.cuda.empty_cache()
