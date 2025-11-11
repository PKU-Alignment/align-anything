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

import gc
from typing import List

import torch
from transformers import AutoConfig

from eval_anything.models.base_model import BaseModel
from eval_anything.utils.data_type import InferenceInput, InferenceOutput


class AccelerateVLAModel(BaseModel):
    def __init__(self, model_cfgs, **kwargs):
        self.model_cfgs = model_cfgs

        self.model_id = self.model_cfgs.model_id
        self.model_name_or_path = self.model_cfgs.model_name_or_path

        self.init_model()

    def get_action_list(self):
        return self.agent.get_action_list()

    def reset(self):
        self.agent.reset()

    def generate(self, observations, goal_spec):
        return self.agent.generate(observations, goal_spec)

    def init_model(self) -> None:
        self.model_config = AutoConfig.from_pretrained(self.model_name_or_path)
        self.ModelForConditionalGeneration = self.model_config.architectures[0]
        self.model_class = globals().get(self.ModelForConditionalGeneration)

        if self.model_class:
            self.model = getattr(self.model_class, 'from_pretrained')(self.model_name_or_path)
        else:
            raise ValueError(f'Model class {self.ModelForConditionalGeneration} not found.')

        self.agent_class = globals().get(self.ModelForConditionalGeneration + 'Agent')

        self.agent = self.agent_class(
            model=self.model,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            sampling='greedy',
            max_seq_len=1000,
        )

        print(f"model '{self.model_id}' loaded successfully.")

    def generation(self, inputs: List[InferenceInput]) -> List[InferenceOutput]:
        return self._generation(inputs)

    def _generation(self, input_list: List[InferenceInput]) -> List[InferenceOutput]:
        inference_outputs = []
        return inference_outputs

    def shutdown_model(self):
        del self.model
        del self.agent
        self.model = None
        self.agent = None
        gc.collect()
        torch.cuda.empty_cache()
