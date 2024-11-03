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

from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    LlavaConfig,
    LlavaForConditionalGeneration,
    LlavaNextConfig,
)


try:
    from transformers import ChameleonConfig
    from align_anything.models.chameleon_model import AccustomedChameleonModel
    CHAMELEON_AVALIABLE = True
except ImportError:
    CHAMELEON_AVALIABLE = False
    print("Chameleon is currently not available.")

try:
    from transformers import Qwen2AudioConfig, Qwen2AudioForConditionalGeneration
    from align_anything.models.qwen2_audio import AccustomedQwen2AudioModel
    Qwen2Audio_AVALIABLE = True
except ImportError:
    Qwen2Audio_AVALIABLE = False
    print("Qwen2Audio is currently not available.")

try:
    from transformers import Qwen2VLConfig, Qwen2VLPreTrainedModel, Qwen2VLForConditionalGeneration
    from align_anything.models.qwen2_vl_model import AccustomedQwen2VLModel
    QWEN2VL_AVALIABLE = True
except ImportError:
    QWEN2VL_AVALIABLE = False
    print("Qwen2VL is currently not available.")

try:
    from align_anything.models.modeling_emu3.mllm.processing_emu3 import Emu3Processor
    EMU_AVAILABLE = True
except ImportError:
    EMU_AVAILABLE = False
    print(f"Emu is currently not available")

from transformers.utils.generic import ModelOutput

from align_anything.models.llava_model import AccustomedLlavaModel
from align_anything.models.llava_next_model import AccustomedLlavaNextModel
from align_anything.models.llama_vision_audio_model import (
    LlamaVisionAudioConfig,
    AccustomedLlamaVisionAudioModel
)


@dataclass
class ScoreModelOutput(ModelOutput):
    """Output of the score model."""

    scores: torch.FloatTensor | None = None  # size = (B, L, D)
    clipped_scores: torch.FloatTensor | None = None  # size = (B, L-I, D)
    end_scores: torch.FloatTensor | None = None  # size = (B, D)
    last_hidden_state: torch.FloatTensor | None = None  # size = (B, L, E)
    clipped_states: torch.FloatTensor | None = None  # size = (B, L-I, D)
    end_last_hidden_state: torch.FloatTensor | None = None  # size = (B, E)
    end_index: torch.LongTensor | None = None  # size = (B,)


def get_score_model(base_pretrained_model, base_llm_model, modality):
    class T2TRewardModel(base_pretrained_model):
        supports_gradient_checkpointing = True

        def __init__(self, config: AutoConfig):
            super().__init__(config)
            setattr(self, self.base_model_prefix, base_llm_model(config))
            self.score_head = nn.Linear(config.hidden_size, 1, bias=False)

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
            B, L, E = last_hidden_state.size()

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

    class TI2TRewardModel(base_pretrained_model):
        supports_gradient_checkpointing = True

        def __init__(self, config: AutoConfig):
            super().__init__(config)
            setattr(self, self.base_model_prefix, base_llm_model(config))
            self.score_head = nn.Linear(5120, 1, bias=False)

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

    class TA2TRewardModel(base_pretrained_model):
        supports_gradient_checkpointing = True

        def __init__(self, config: AutoConfig):
            super().__init__(config)
            setattr(self, self.base_model_prefix, base_llm_model(config))
            self.score_head = nn.Linear(4096, 1, bias=False)

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
            
    class TV2TRewardModel(Qwen2VLForConditionalGeneration):
        supports_gradient_checkpointing = True

        def __init__(self, config: AutoConfig):
            super().__init__(config)
            self.score_head = nn.Linear(3584, 1, bias=False) 
            # Note that if you are dealing with Qwen2-VL-2B, you should change 3584 to 1536. For Qwen2-VL-72B, you should change 3584 to 8192

        def forward(
            self,
            input_ids: torch.LongTensor | None = None,
            attention_mask: torch.Tensor | None = None,
            **kwargs,
        ) -> torch.Tensor:
            outputs = super().forward(
                input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                **kwargs,
            )

            last_hidden_state = outputs.hidden_states[-1]
            scores = self.score_head(last_hidden_state).float()
            B, L, E = last_hidden_state.size()

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
            
    if modality == 'text':
        RewardModel = T2TRewardModel
    elif modality == 'text_image_to_text':
        RewardModel = TI2TRewardModel
    elif modality == 'text_audio_to_text':
        RewardModel = TA2TRewardModel
    elif modality == 'text_video_to_text':
        RewardModel = TV2TRewardModel

    return RewardModel


class AnyBaseModelCLS(AutoModel):
    """Any model for base class."""


def register_model(auto_model: AutoModelForCausalLM) -> AutoModelForCausalLM:
    auto_model.register(LlavaConfig, LlavaForConditionalGeneration)
    auto_model.register(LlavaNextConfig, AccustomedLlavaNextModel)
    auto_model.register(LlamaVisionAudioConfig, AccustomedLlamaVisionAudioModel)
    if Qwen2Audio_AVALIABLE:
        auto_model.register(Qwen2AudioConfig, Qwen2AudioForConditionalGeneration)
    if CHAMELEON_AVALIABLE:
        auto_model.register(ChameleonConfig, AccustomedChameleonModel)
    if QWEN2VL_AVALIABLE:
        auto_model.register(Qwen2VLConfig, AccustomedQwen2VLModel)
    return auto_model


def register_base_model(auto_model: AnyBaseModelCLS) -> AnyBaseModelCLS:
    auto_model.register(LlavaConfig, AccustomedLlavaModel)
    auto_model.register(LlavaNextConfig, AccustomedLlavaNextModel)
    auto_model.register(LlamaVisionAudioConfig, AccustomedLlamaVisionAudioModel)
    if Qwen2Audio_AVALIABLE:
        auto_model.register(Qwen2AudioConfig, AccustomedQwen2AudioModel)
    return auto_model


AnyModel = register_model(AutoModelForCausalLM)
AnyBaseModel = register_base_model(AnyBaseModelCLS)
