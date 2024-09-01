# Copyright 2024 PKU-Alignment Team and tatsu-lab. All Rights Reserved.
#
# This code is inspired by the tgxs002's HPSv2 library.
# https://github.com/tgxs002/HPSv2/blob/master/hpsv2/src/open_clip/factory.py
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

import json
import logging
import os
import pathlib
import re
from copy import deepcopy
from pathlib import Path
from turtle import forward
from typing import Any, Dict, Optional, Tuple, Union
import torch
from .model import CLIP, convert_to_custom_text_state_dict, resize_pos_embed
from .pretrained import get_pretrained_cfg, download_pretrained, list_pretrained_tags_by_model
from .transform import image_transform, OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
from .tokenizer import HFTokenizer, tokenize


HF_HUB_PREFIX = 'hf-hub:'
_MODEL_CONFIG_PATHS = [Path(__file__).parent]
_MODEL_CONFIGS = {}


def _natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_.lower())]


def _rescan_model_configs():
    global _MODEL_CONFIGS

    config_ext = ('.json',)
    config_files = []
    for config_path in _MODEL_CONFIG_PATHS:
        if config_path.is_file() and config_path.suffix in config_ext:
            config_files.append(config_path)
        elif config_path.is_dir():
            for ext in config_ext:
                config_files.extend(config_path.glob(f'*{ext}'))

    for cf in config_files:
        with open(cf, 'r') as f:
            model_cfg = json.load(f)
            if all(a in model_cfg for a in ('embed_dim', 'vision_cfg', 'text_cfg')):
                _MODEL_CONFIGS[cf.stem] = model_cfg

    _MODEL_CONFIGS = {k: v for k, v in sorted(_MODEL_CONFIGS.items(), key=lambda x: _natural_key(x[0]))}

_rescan_model_configs()

def list_models():
    return list(_MODEL_CONFIGS.keys())

def get_model_config(model_name):
    if model_name in _MODEL_CONFIGS:
        return deepcopy(_MODEL_CONFIGS[model_name])
    else:
        return None

def get_tokenizer(model_name):
    if model_name.startswith(HF_HUB_PREFIX):
        tokenizer = HFTokenizer(model_name[len(HF_HUB_PREFIX):])
    else:
        config = get_model_config(model_name)
        tokenizer = HFTokenizer(
            config['text_cfg']['hf_tokenizer_name']) if 'hf_tokenizer_name' in config['text_cfg'] else tokenize
    return tokenizer


def load_state_dict(checkpoint_path: str, map_location='cpu'):
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    if next(iter(state_dict.items()))[0].startswith('module'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    return state_dict


def load_checkpoint(model, checkpoint_path, strict=True):
    state_dict = load_state_dict(checkpoint_path)
    if 'positional_embedding' in state_dict and not hasattr(model, 'positional_embedding'):
        state_dict = convert_to_custom_text_state_dict(state_dict)
    resize_pos_embed(state_dict, model)
    incompatible_keys = model.load_state_dict(state_dict, strict=strict)
    return incompatible_keys


def create_model(
        model_name: str,
        pretrained: Optional[str] = None,
        device: Union[str, torch.device] = 'cpu',
        output_dict: Optional[bool] = None,
):
    model_name = model_name.replace('/', '-')
    checkpoint_path = None
    pretrained_cfg = {}

    if isinstance(device, str):
        device = torch.device(device)

    model_cfg = get_model_config(model_name)
    if model_cfg is not None:
        logging.info(f'Loaded {model_name} model config.')
    else:
        logging.error(f'Model config for {model_name} not found; available models {list_models()}.')
        raise RuntimeError(f'Model config for {model_name} not found.')

    model = CLIP(**model_cfg, cast_dtype=None)

    checkpoint_path = ''
    pretrained_cfg = get_pretrained_cfg(model_name, pretrained)
    if pretrained_cfg:
        checkpoint_path = download_pretrained(pretrained_cfg, cache_dir=None)
    elif os.path.exists(pretrained):
        checkpoint_path = pretrained

    if checkpoint_path:
        logging.info(f'Loading pretrained {model_name} weights ({pretrained}).')
        load_checkpoint(model, checkpoint_path)
    else:
        error_str = (
            f'Pretrained weights ({pretrained}) not found for model {model_name}.'
            f'Available pretrained tags ({list_pretrained_tags_by_model(model_name)}.')
        logging.warning(error_str)
        raise RuntimeError(error_str)
    
    model.to(device=device)

    model.visual.image_mean = pretrained_cfg.get('mean', None) or OPENAI_DATASET_MEAN
    model.visual.image_std = pretrained_cfg.get('std', None) or OPENAI_DATASET_STD

    if output_dict and hasattr(model, "output_dict"):
        model.output_dict = True

    return model

def create_model_and_transforms(
        model_name: str,
        pretrained: Optional[str] = None,
        device: Union[str, torch.device] = 'cpu',
        output_dict: Optional[bool] = None
):
    model = create_model(
        model_name,
        pretrained,
        device=device,
        output_dict=output_dict,
    )

    image_mean = getattr(model.visual, 'image_mean', None)
    image_std = getattr(model.visual, 'image_std', None)

    preprocess_val = image_transform(
        model.visual.image_size,
        mean=image_mean,
        std=image_std,
        resize_longest_max=True,
    )

    return model, preprocess_val