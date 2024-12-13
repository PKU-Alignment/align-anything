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


import torch
import torch.utils.checkpoint
import torch.nn as nn
from transformers import LlavaPreTrainedModel, AutoConfig
from align_anything.models.reward_model import ScoreModelOutput

from transformers.models.llava.modeling_llava import (
    LlavaForConditionalGeneration,
)

class AccustomedLlavaModel(LlavaForConditionalGeneration):

    @property
    def infer_required_keys(self) -> list[str]:
        return ['input_ids', 'attention_mask', 'pixel_values', 'labels']
    
    @property
    def processor_available(self):
        return True

    @property
    def forbidden_keys(self) -> list[str]:
        return ['images', 'response_lens']

    def infer_batch(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Return the dict used for model inference"""
        new_batch = {}
        for key, value in batch.items():
            if key not in self.forbidden_keys:
                new_batch[key] = value
        return new_batch
    
class AccustomedLlavaRewardModel(LlavaPreTrainedModel):

    supports_gradient_checkpointing = True

    def __init__(self, config: AutoConfig):
        super().__init__(config)
        setattr(self, self.base_model_prefix, AccustomedLlavaModel(config))
        self.score_head = nn.Linear(self.model.language_model.lm_head.in_features, 1, bias=False)

    @property
    def infer_required_keys(self) -> list[str]:
        return ['input_ids', 'attention_mask', 'pixel_values']
    
    @property
    def processor_available(self):
        return True

    def infer_batch(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Return the dict used for model inference"""
        return {
            'input_ids': batch['input_ids'],
            'attention_mask': batch['attention_mask'],
            'pixel_values': batch['pixel_values'],
        }

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            **kwargs,
        )

        last_hidden_state = outputs.hidden_states[-1]
        scores = self.score_head(last_hidden_state).float()
        B, _, _ = scores.size()

        end_index = -torch.ones((B,))  # size = (B,)
        end_last_hidden_state = last_hidden_state[:, -1, :].unsqueeze(1)
        end_scores = self.score_head(end_last_hidden_state).float()
        end_last_hidden_state = end_last_hidden_state.squeeze(dim=1)  # size = (B, E)
        end_scores = end_scores.squeeze(dim=1)  # size = (B, D)

        return ScoreModelOutput(
            scores=scores,  # size = (B, L, D)
            end_scores=end_scores,  # size = (B, D)
            last_hidden_state=last_hidden_state,  # size = (B, L, E)
            end_last_hidden_state=end_last_hidden_state,  # size = (B, E)
            end_index=end_index,  # size = (B,)
        )