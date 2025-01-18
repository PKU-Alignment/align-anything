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
from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLForConditionalGeneration
from transformers.utils import is_torchdynamo_compiling
from transformers.cache_utils import StaticCache

from align_anything.models.reward_model import ScoreModelOutput


class AccustomedQwen2VLModel(Qwen2VLForConditionalGeneration):

    @property
    def processor_available(self):
        return True

    @property
    def embed_tokens(self):
        return self.model.embed_tokens


class AccustomedQwen2VLRewardModel(Qwen2VLForConditionalGeneration):

    supports_gradient_checkpointing = True

    def __init__(self, config: AutoConfig):
        super().__init__(config)
        self.score_head = nn.Linear(3584, 1, bias=False)

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

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        **kwargs,
    ):
        # Overwritten -- in specific circumstances we don't want to forward image inputs to the model

        # If we have cache: let's slice `input_ids` through `cache_position`, to keep only the unprocessed tokens
        # Exception 1: when passing input_embeds, input_ids may be missing entries
        # Exception 2: some generation methods do special slicing of input_ids, so we don't need to do it here
        if past_key_values is not None:
            if (
                inputs_embeds is not None  # Exception 1
                or (is_torchdynamo_compiling() or cache_position[-1] >= input_ids.shape[1])  # Exception 3
            ):
                input_ids = input_ids[:, -cache_position.shape[0] :]
            elif input_ids.shape[1] != cache_position.shape[0]:  # Default case (the "else", a no op, is Exception 2)
                input_ids = input_ids[:, cache_position]

        if cache_position[0] != 0:
            pixel_values = None
            pixel_values_videos = None

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and cache_position[0] == 0:
            model_inputs = {"inputs_embeds": inputs_embeds, "input_ids": None}
        else:
            model_inputs = {"input_ids": input_ids, "inputs_embeds": None}

        if isinstance(past_key_values, StaticCache) and attention_mask.ndim == 2:
            if model_inputs["inputs_embeds"] is not None:
                batch_size, sequence_length, _ = inputs_embeds.shape
                device = inputs_embeds.device
            else:
                batch_size, sequence_length = input_ids.shape
                device = input_ids.device

            attention_mask = self.model._prepare_4d_causal_attention_mask_with_cache_position(
                attention_mask,
                sequence_length=sequence_length,
                target_length=past_key_values.get_max_cache_shape(),
                dtype=self.lm_head.weight.dtype,
                device=device,
                cache_position=cache_position,
                batch_size=batch_size,
                config=self.config,
                past_key_values=past_key_values,
            )

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
                "pixel_values_videos": pixel_values_videos,
                "image_grid_thw": image_grid_thw,
                "video_grid_thw": video_grid_thw,
                "cache_position": cache_position,
            }
        )
        return model_inputs