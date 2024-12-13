# Copyright 2024 PKU-Alignment Team. and Meta Inc. and The HuggingFace Inc. team. All Rights Reserved.
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


from typing import List, Optional, Tuple, Union, Any

import torch
import torch.utils.checkpoint
from transformers import AutoConfig
from transformers.cache_utils import Cache
from transformers.modeling_outputs import (
    CausalLMOutputWithPast,
)
from transformers.models.chameleon.modeling_chameleon import (
    ChameleonForConditionalGeneration,
)
from torch.nn import CrossEntropyLoss
from torch import nn

from align_anything.models.reward_model import ScoreModelOutput

class AccustomedChameleonModel(ChameleonForConditionalGeneration):

    def pre_tokenization(
        self, 
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
    ):
        if pixel_values is None:
            
            return_dict = {
                "input_ids": input_ids,
            }
            return return_dict
        image_tokens = self.model.get_image_tokens(pixel_values)
        special_image_mask = input_ids == self.model.vocabulary_mapping.image_token_id
        image_tokens = image_tokens.to(input_ids.device, input_ids.dtype)
        input_ids = input_ids.masked_scatter(special_image_mask, image_tokens)
        
        return_dict =  {
            "input_ids": input_ids.to("cpu"),
            "pixel_values": pixel_values.to("cpu"),
        }
        
        return return_dict
        
    @property
    def processor_available(self):
        return True

    def apply_chat_template(self, 
                            messages: list[dict[str, Any]], 
                            add_generation_prompt: bool =False) -> dict[str, Any]:
        # use default format
        final_text = ''
        for line in messages:
            for content in line['content']:
                if content['type'] == 'text':
                    final_text += content['text']
        return final_text
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import ChameleonProcessor, ChameleonForConditionalGeneration
        >>> import torch
        >>> import requests
        >>> from PIL import Image

        >>> model = ChameleonForConditionalGeneration.from_pretrained("facebook/chameleon-7b", torch_dtype=torch.bfloat16)
        >>> processor = ChameleonProcessor.from_pretrained("facebook/chameleon-7b")

        >>> prompt = "I used to know a lot about constellations when I was younger, but as I grew older, I forgot most of what I knew. These are the only two constellations that I really remember now.<image><image>I would like for you to tell me about 3 more constellations and give me a little bit of history about the constellation."
        >>> image = Image.open(requests.get("https://nineplanets.org/wp-content/uploads/2020/12/the-big-dipper-1.jpg", stream=True).raw)
        >>> image_2 = Image.open(requests.get("https://www.kxan.com/wp-content/uploads/sites/40/2020/10/ORION.jpg", stream=True).raw)

        >>> inputs = processor(prompt, images=[image, image_2], return_tensors="pt").to(model.device, torch.bfloat16)

        >>> generated_ids = model.generate(**inputs, max_new_tokens=100, do_sample=False)
        >>> processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        ```"""
        
        # print(f"pixel value dtype: {pixel_values.dtype}")
        # pixel_values = pixel_values.to(dtype = torch.bfloat16)
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        torch.set_printoptions(threshold=torch.inf)
        # print(f"input_ids: {input_ids}")
        # print(f"Labels: {labels}")
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class AccustomedChameleonRewardModel(ChameleonForConditionalGeneration):
    
    supports_gradient_checkpointing = True

    def __init__(self, config: AutoConfig):
        super().__init__(config)
        setattr(self, self.base_model_prefix, AccustomedChameleonModel(config))
        self.score_head = nn.Linear(4096, 1, bias=False)

    @property
    def infer_required_keys(self) -> list[str]:
        return ['input_ids', 'attention_mask']

    def infer_batch(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {
            'input_ids': batch['input_ids'],
            'attention_mask': batch['attention_mask'],
        }

    def forward(self,
                input_ids: torch.LongTensor, 
                attention_mask: torch.LongTensor, 
                **kwargs
    ) -> torch.FloatTensor:
        outputs = super().forward(input_ids, attention_mask, **kwargs)
        hidden_states = outputs[0]
        scores = self.score_head(hidden_states)
        end_scores = scores[:, -1, :]
        return ScoreModelOutput(
            scores=scores,
            end_scores=end_scores,
            last_hidden_state=hidden_states,
        )