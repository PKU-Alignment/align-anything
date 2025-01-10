# Copyright 2024 PKU-Alignment Team. All Rights Reserved.
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
from transformers import AutoConfig, Idefics2PreTrainedModel
from transformers.models.idefics2.modeling_idefics2 import Idefics2ForConditionalGeneration

from align_anything.models.reward_model import ScoreModelOutput


class AccustomedIdefics2Model(Idefics2ForConditionalGeneration):
    """Accustomed Interface for Idefics2 model"""

    @property
    def chat_template(self):
        return "{% for message in messages %}{{message['role'].capitalize()}}{% if message['content'][0]['type'] == 'image' %}{{':'}}{% else %}{{': '}}{% endif %}{% for line in message['content'] %}{% if line['type'] == 'text' %}{{line['text']}}{% elif line['type'] == 'image' %}{{ '<image>' }}{% endif %}{% endfor %}<end_of_utterance>\n{% endfor %}{% if add_generation_prompt %}{{ 'Assistant:' }}{% endif %}"


class AccustomedIdefics2RewardModel(Idefics2PreTrainedModel):
    """Accustomed Interface for Idefics2 reward model"""

    supports_gradient_checkpointing = True

    def __init__(self, config: AutoConfig):
        super().__init__(config)
        setattr(self, self.base_model_prefix, AccustomedIdefics2Model(config))
        self.score_head = nn.Linear(self.model.language_model.lm_head.in_features, 1, bias=False)

    @property
    def processor_available(self):
        return True

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
