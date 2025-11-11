# Copyright 2025 PKU-Alignment Team. All Rights Reserved.
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
import torch.nn as nn
import torch.utils.checkpoint
from transformers import AutoConfig, Qwen2Model, Qwen2PreTrainedModel
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM

from align_anything.models.reward_model import ScoreModelOutput


class AccustomedQwen2Model(Qwen2ForCausalLM):
    """Accustomed Interface for Qwen2 model"""

    @property
    def processor_available(self):
        return False


class AccustomedQwen2RewardModel(Qwen2PreTrainedModel):

    supports_gradient_checkpointing = True

    def __init__(self, config: AutoConfig):
        super().__init__(config)
        setattr(self, self.base_model_prefix, Qwen2Model(config))
        self.score_head = nn.Linear(config.hidden_size, 1, bias=False)

    @property
    def processor_available(self):
        return False

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
        B, L, _ = last_hidden_state.size()

        if attention_mask is None:
            if B > 1:
                raise ValueError("'attention_mask' is required when batch size > 1.")
            attention_mask = last_hidden_state.new_ones(B, L, dtype=torch.bool)  # size = (B, L)

        end_index = torch.cat([m.nonzero()[-1] for m in attention_mask])  # size = (B,)
        end_last_hidden_state = torch.gather(  # size = (B, 1, E)
            last_hidden_state,
            dim=1,
            index=(
                end_index.to(last_hidden_state.device)
                .unsqueeze(dim=1)
                .unsqueeze(dim=2)
                .expand(-1, -1, last_hidden_state.size(-1))
            ),
        )
        end_scores = torch.gather(  # size = (B, 1, D)
            scores,
            dim=1,
            index=(
                end_index.to(scores.device)
                .unsqueeze(dim=1)
                .unsqueeze(dim=2)
                .expand(-1, -1, scores.size(-1))
            ),
        )
        end_last_hidden_state = end_last_hidden_state.squeeze(dim=1)  # size = (B, E)
        end_scores = end_scores.squeeze(dim=1)  # size = (B, D)

        return ScoreModelOutput(
            scores=scores,  # size = (B, L, D)
            end_scores=end_scores,  # size = (B, D)
            last_hidden_state=last_hidden_state,  # size = (B, L, E)
            end_last_hidden_state=end_last_hidden_state,  # size = (B, E)
            end_index=end_index,  # size = (B,)
        )
