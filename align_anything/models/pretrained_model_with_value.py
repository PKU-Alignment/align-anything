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

import os
from typing import Any, Literal

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoProcessor, AutoTokenizer

from align_anything.models.model_registry import AnyBaseModel, get_score_model
from align_anything.models.pretrained_model import resize_tokenizer_embedding


def load_pretrained_model_with_value_head(
    model_name_or_path: str | os.PathLike,
    model_max_length: int = 512,
    padding_side: Literal['left', 'right'] = 'right',
    auto_device_mapping: bool = False,
    freeze_vision_tower: bool = True,
    freeze_audio_tower: bool = True,
    freeze_mm_proj: bool = True,
    freeze_vision_proj: bool = True,
    freeze_audio_proj: bool = True,
    freeze_language_model: bool = False,
    dtype: torch.dtype | str | None = 'auto',
    *,
    cache_dir: str | os.PathLike | None = None,
    trust_remote_code: bool = False,
    auto_model_args: tuple[Any, ...] = (),
    auto_model_kwargs: dict[str, Any] | None = None,
    auto_tokenizer_args: tuple[Any, ...] = (),
    auto_tokenizer_kwargs: dict[str, Any] | None = None,
    modality: str = 'text'
) -> nn.Module:
    model_name_or_path = os.path.expanduser(model_name_or_path)
    cache_dir = os.path.expanduser(cache_dir) if cache_dir is not None else None
    device_map = 'auto' if auto_device_mapping else None
    if auto_model_kwargs is None:
        auto_model_kwargs = {}
    if auto_tokenizer_kwargs is None:
        auto_tokenizer_kwargs = {}

    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    base_class = AnyBaseModel._model_mapping[type(config)]

    try:
        base_pretrained_class = base_class.pretrain_class()
    except AttributeError:
        base_pretrained_class = base_class.__base__

    AnyRewardModel = get_score_model(base_pretrained_class, base_class, modality)
    ignore_mismatched_sizes = modality in ['text_audio_to_text']
    model = AnyRewardModel.from_pretrained(
        model_name_or_path,
        *auto_model_args,
        cache_dir=cache_dir,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
        ignore_mismatched_sizes=ignore_mismatched_sizes,
        **auto_model_kwargs,
    )

    forbidden_modules = set()
    if freeze_vision_tower:
        forbidden_modules.add('vision_tower')
    if freeze_audio_tower:
        forbidden_modules.add('audio_tower')
    if freeze_mm_proj:
        forbidden_modules.add('multi_modal_projector')
    if freeze_vision_proj:
        forbidden_modules.add('image_projector')
    if freeze_audio_proj:
        forbidden_modules.add('audio_projector')
    if freeze_language_model:
        forbidden_modules.add('language_model')
    for name, param in model.named_parameters():
        if not any(forbidden_module in name for forbidden_module in forbidden_modules):
            if dtype == torch.float32:
                param.data = param.data.to(torch.float32)
        else:
            param.requires_grad_(False)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        *auto_tokenizer_args,
        cache_dir=cache_dir,
        model_max_length=model_max_length,
        padding_side=padding_side,
        trust_remote_code=trust_remote_code,
        **auto_tokenizer_kwargs,
    )

    # MoE - balancing loss
    model_config = model.config.to_dict()
    if 'output_router_logits' in model_config:
        print('[MoE] set output_router_logits as True')
        model.config.output_router_logits = True

    try:
        processor = AutoProcessor.from_pretrained(model_name_or_path)
        processor.tokenizer.padding_side = padding_side
        resize_tokenizer_embedding(tokenizer=processor.tokenizer, model=model)

        return model, processor.tokenizer, processor
    except Exception:
        processor = None
        resize_tokenizer_embedding(tokenizer=tokenizer, model=model)
        
        return model, tokenizer, processor
