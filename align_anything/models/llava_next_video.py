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
from torch import nn
from transformers import AutoConfig, LlavaNextVideoForConditionalGeneration

from align_anything.models.reward_model import ScoreModelOutput


class AccustomedLlavaNextVideoModel(LlavaNextVideoForConditionalGeneration):
    """Accustomed Interface for LlavaNext model"""

    @property
    def processor_available(self):
        return True


class AccustomedLlavaNextVideoRewardModel(LlavaNextVideoForConditionalGeneration):
    """Accustomed Interface for LlavaNextVideo model"""

    supports_gradient_checkpointing = True

    def __init__(self, config: AutoConfig):
        super().__init__(config)
        self.score_head = nn.Linear(4096, 1, bias=False)

    @property
    def processor_available(self):
        return True

    def forward(
        self,
        **kwargs,
    ) -> ScoreModelOutput:
        outputs = super().forward(**kwargs, output_hidden_states=True)
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
