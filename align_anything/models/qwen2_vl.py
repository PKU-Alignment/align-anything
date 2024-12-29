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
from torch import nn
from transformers import AutoConfig
from transformers.models.qwen2_vl.modeling_qwen2_vl import (
    Qwen2VLForConditionalGeneration,
)

from align_anything.models.reward_model import ScoreModelOutput


class AccustomedQwen2VLModel(Qwen2VLForConditionalGeneration):
    """Accustomed Interface for Qwen2VL model"""

    @property
    def processor_available(self):
        return True


class AccustomedQwen2VLRewardModel(Qwen2VLForConditionalGeneration):

    supports_gradient_checkpointing = True

    def __init__(self, config: AutoConfig):
        super().__init__(config)
        setattr(self, self.base_model_prefix, AccustomedQwen2VLModel(config))
        self.score_head = nn.Linear(3584, 1, bias=False)

    def infer_batch(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {
            'input_ids': batch['input_ids'],
            'attention_mask': batch['attention_mask'],
            'pixel_values': batch['pixel_values'],
            'pixel_values_videos': batch['pixel_values_videos'],
        }

    @property
    def processor_available(self):
        return True

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        pixel_values_videos: torch.FloatTensor,
        **kwargs,
    ) -> ScoreModelOutput:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            **kwargs,
        )
        last_hidden_state = outputs.hidden_states[-1]
        scores = self.score_head(last_hidden_state)
        B, _, _ = scores.size()
        end_index = -torch.ones((B,))
        end_last_hidden_state = last_hidden_state[:, -1, :].unsqueeze(1)
        end_scores = self.score_head(end_last_hidden_state).float()
        end_last_hidden_state = end_last_hidden_state.squeeze(dim=1)
        end_scores = end_scores.squeeze(dim=1)

        return ScoreModelOutput(
            scores=scores,
            end_scores=end_scores,
            last_hidden_state=last_hidden_state,
            end_last_hidden_state=end_last_hidden_state,
            end_index=end_index,
        )
