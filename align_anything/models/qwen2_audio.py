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

import deepspeed
import torch
import torch.utils.checkpoint
import torch.nn as nn
from transformers import AutoConfig

from align_anything.models.reward_model import ScoreModelOutput

try:
    from transformers import Qwen2AudioForConditionalGeneration, Qwen2AudioPreTrainedModel
    Qwen2Audio_AVALIABLE = True

except ImportError:
    Qwen2Audio_AVALIABLE = False
    print("Qwen2Audio is currently not available.")

if Qwen2Audio_AVALIABLE:
    class AccustomedQwen2AudioModel(Qwen2AudioForConditionalGeneration):
        
        def __init__(self, config: AutoConfig):
            super().__init__(config)
            deepspeed.zero.register_external_parameter(self, self.audio_tower.embed_positions.weight)

        @property
        def processor_available(self):
            return True

        @property
        def infer_required_keys(self) -> list[str]:
            return ['input_ids', 'attention_mask', 'input_features', 'feature_attention_mask', 'labels']
        
        def infer_batch(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
            """Return the dict used for model inference"""
            return {
                'input_ids': batch['input_ids'],
                'attention_mask': batch['attention_mask'],
                'input_features': batch['input_features'],
                'feature_attention_mask': batch['feature_attention_mask'],
                'labels': batch.get('labels'),
            }

    class AccustomedQwen2AudioRewardModel(Qwen2AudioPreTrainedModel):
        supports_gradient_checkpointing = True

        def __init__(self, config: AutoConfig):
            super().__init__(config)
            setattr(self, self.base_model_prefix, AccustomedQwen2AudioModel(config))
            deepspeed.zero.register_external_parameter(self.model, self.model.audio_tower.embed_positions.weight)
            self.score_head = nn.Linear(4096, 1, bias=False)

        @property
        def infer_required_keys(self) -> list[str]:
            return ['input_ids', 'attention_mask', 'input_features', 'feature_attention_mask']
        
        @property
        def processor_available(self):
            return True

        def infer_batch(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
            """Return the dict used for model inference"""
            return {
                'input_ids': batch['input_ids'],
                'attention_mask': batch['attention_mask'],
                'input_features': batch['input_features'],
                'feature_attention_mask': batch['feature_attention_mask'],
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
            final_attention_mask = outputs.attention_mask

            last_hidden_state = outputs.hidden_states[-1]
            scores = self.score_head(last_hidden_state).float()

            end_index = torch.cat([m.nonzero()[-1] for m in final_attention_mask])  # size = (B,)
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
