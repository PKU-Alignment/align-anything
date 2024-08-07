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

from __future__ import annotations

import argparse
import json
import os
import random
from collections import namedtuple
from typing import Any, NamedTuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.nn.utils.rnn import pad_sequence
from torch.types import Number
from transformers import PreTrainedTokenizerBase, ProcessorMixin
from transformers.tokenization_utils import BatchEncoding, PaddingStrategy, TruncationStrategy


def right_padding(sequences: list[torch.Tensor], padding_value: Number) -> torch.Tensor:
    return pad_sequence(sequences, batch_first=True, padding_value=padding_value)


def left_padding(sequences: list[torch.Tensor], padding_value: Number) -> torch.Tensor:
    return right_padding(
        [seq.flip(0) for seq in sequences],
        padding_value=padding_value,
    ).flip(1)


def dict_to_namedtuple(dic):
    def convert(value):
        if isinstance(value, dict):
            return dict_to_namedtuple(value)
        elif isinstance(value, list):
            return [convert(item) for item in value]
        else:
            return value

    class EnhancedNamedTuple(namedtuple('configs', dic.keys())):
        __slots__ = ()

        def __getattr__(self, item):
            return None

    cfgs = EnhancedNamedTuple(**{k: convert(v) for k, v in dic.items()})
    return cfgs


def namedtuple_to_dict(obj: Any) -> Any:
    if isinstance(obj, tuple) and hasattr(obj, '_fields'):
        return {field: namedtuple_to_dict(getattr(obj, field)) for field in obj._fields}
    elif isinstance(obj, list):
        return [namedtuple_to_dict(item) for item in obj]
    else:
        return obj

def requestoutput_to_dict(data, mode='brief'):
    if mode == 'brief':
        info = {
            "prompt": data.prompt,
            "outputs": []
        }
    else:
        info = {
            "prompt": data.prompt,
            "prompt_token_ids": data.prompt_token_ids,
            "prompt_logprobs": [vllm_logprob_to_dict(token_logprob) for token_logprob in data.prompt_logprobs[1:]],
            'outputs': [],
            'finished': data.finished,
            'metrics':
            {
                'arrival_time': data.metrics.arrival_time,
                'last_token_time': data.metrics.last_token_time,
                'first_scheduled_time': data.metrics.first_scheduled_time,
                'first_token_time': data.metrics.first_token_time,
                'time_in_queue': data.metrics.time_in_queue,
                'finished_time': data.metrics.finished_time,
            }
        }
    for output in data.outputs:
        if mode == 'brief':
            output = {
                'index': output.index,
                'text': output.text,
            }
        else:
            output = {
                'index': output.index,
                'text': output.text,
                'token_ids': output.token_ids,
                'cumulative_logprob': output.cumulative_logprob,
                'logprobs': [vllm_logprob_to_dict(token_logprob) for token_logprob in output.logprobs],
                'finish_reason': output.finish_reason,
                'stop_reason': output.stop_reason
            }
        info['outputs'].append(output)
        return info

def vllm_logprob_to_dict(data):
    # print([{v.decoded_token: v.logprob} for k, v in data.items()])
    return [{v.decoded_token: v.logprob} for k, v in data.items()]

def read_cfgs(mode: str, task: str) -> list[dict[str, Any], dict[str, Any]]:
    current_file_path = os.path.abspath(__file__)
    parent_path = os.path.dirname(os.path.dirname(current_file_path))
    yaml_path = os.path.join(parent_path, 'configs', mode, f'{task}.yaml')
    with open(yaml_path, encoding='utf-8') as f:
        try:
            configs = yaml.safe_load(f)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f'{yaml_path} error: {exc}') from exc
    ds_cfgs_path = os.path.join(
        parent_path,
        'configs',
        'deepspeed',
        configs['train_cfgs']['ds_cfgs'],
    )
    with open(ds_cfgs_path) as f:
        ds_cfgs = json.load(f)

    return configs, ds_cfgs

def read_eval_cfgs(task: str, backend: str) -> dict[str, Any]:
    current_file_path = os.path.abspath(__file__)
    parent_path = os.path.dirname(os.path.dirname(current_file_path))
    yaml_path = os.path.join(parent_path, 'configs', 'evaluation', 'benchmarks', f'{task}.yaml')
    with open(yaml_path, encoding='utf-8') as f:
        try:
            configs = yaml.safe_load(f)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f'{yaml_path} error: {exc}') from exc
    if backend.lower() == 'vllm':
        infer_cfgs_path = os.path.join(
            parent_path,
            'configs',
            'evaluation',
            'vllm',
            configs['infer_cfgs']['vllm_cfgs'],
        )
    else:
        infer_cfgs_path = os.path.join(
            parent_path,
            'configs',
            'evaluation',
            'deepspeed',
            configs['infer_cfgs']['ds_cfgs'],
        )
    with open(infer_cfgs_path) as f:
        infer_cfgs = json.load(f)

    return configs, infer_cfgs

def get_optimizer_grouped_parameters(
    module: nn.Module,
    weight_decay: float,
    no_decay_name_set: set[str] | None = None,
) -> list[dict[str, list[nn.Parameter] | float]]:
    """Get parameter groups with customized weight decay value."""
    if no_decay_name_set is None:
        no_decay_name_set = {'bias', 'LayerNorm.weight'}
    no_decay_name_set = set(map(str.lower, no_decay_name_set))

    named_parameters = [
        (name.lower(), param) for name, param in module.named_parameters() if param.requires_grad
    ]

    return [
        {
            'params': [
                param
                for name, param in named_parameters
                if not any(no_decay_name in name for no_decay_name in no_decay_name_set)
            ],
            'weight_decay': weight_decay,
        },
        {
            'params': [
                param
                for name, param in named_parameters
                if any(no_decay_name in name for no_decay_name in no_decay_name_set)
            ],
            'weight_decay': 0.0,
        },
    ]


def prepare_ds_train_cfgs(custom_cfgs: NamedTuple, raw_ds_cfgs: dict[str, Any]) -> dict[str, Any]:
    """Prepare the DeepSpeed config for training."""
    ds_cfgs = raw_ds_cfgs.copy()
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    micro_batch_size_per_gpu = custom_cfgs.per_device_train_batch_size
    gradient_accumulation_steps = custom_cfgs.gradient_accumulation_steps

    train_batch_size = micro_batch_size_per_gpu * world_size * gradient_accumulation_steps
    ds_cfgs['train_batch_size'] = train_batch_size
    ds_cfgs['train_micro_batch_size_per_gpu'] = micro_batch_size_per_gpu
    ds_cfgs['gradient_accumulation_steps'] = gradient_accumulation_steps

    ds_cfgs['bf16']['enabled'] = custom_cfgs.bf16
    ds_cfgs['fp16']['enabled'] = custom_cfgs.fp16
    return ds_cfgs


def prepare_accelerate_train_cfgs(custom_cfgs: NamedTuple) -> dict[str, Any]:
    """Prepare the DeepSpeed config for training."""
    cfgs = {}
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    micro_batch_size_per_gpu = custom_cfgs.per_device_train_batch_size
    gradient_accumulation_steps = custom_cfgs.gradient_accumulation_steps

    train_batch_size = micro_batch_size_per_gpu * world_size * gradient_accumulation_steps
    cfgs['train_batch_size'] = train_batch_size
    cfgs['train_micro_batch_size_per_gpu'] = micro_batch_size_per_gpu
    cfgs['gradient_accumulation_steps'] = gradient_accumulation_steps

    if custom_cfgs.bf16:
        mixed_precision = 'bf16'
    elif custom_cfgs.fp16:
        mixed_precision = 'fp16'
    else:
        mixed_precision = 'no'

    cfgs['mixed_precision'] = mixed_precision
    return cfgs


def prepare_ds_eval_cfgs(custom_cfgs: NamedTuple, raw_ds_cfgs: dict[str, Any]) -> dict[str, Any]:
    """Prepare the DeepSpeed config for training."""
    ds_cfgs = raw_ds_cfgs.copy()
    # The evaluation config only works for ZeRO stage 0 and ZeRO stage 3
    if ds_cfgs['zero_optimization']['stage'] in {1, 2}:
        ds_cfgs['zero_optimization']['stage'] = 0

    ds_cfgs['train_batch_size'] = None
    ds_cfgs['train_micro_batch_size_per_gpu'] = 1
    ds_cfgs['gradient_accumulation_steps'] = 1

    ds_cfgs['bf16']['enabled'] = custom_cfgs.bf16
    ds_cfgs['fp16']['enabled'] = custom_cfgs.fp16
    return ds_cfgs


def update_dict(total_dict: dict[str, Any], item_dict: dict[str, Any]) -> dict[str, Any]:
    def update_dict(total_dict: dict[str, Any], item_dict: dict[str, Any]) -> dict[str, Any]:
        for key, value in total_dict.items():
            if key in item_dict:
                total_dict[key] = item_dict[key]
            if isinstance(value, dict):
                update_dict(value, item_dict)
        return total_dict

    return update_dict(total_dict, item_dict)


def is_convertible_to_float(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def custom_cfgs_to_dict(key_list: str, value: Any) -> dict[str, Any]:
    """This function is used to convert the custom configurations to dict."""
    if value == 'True':
        value = True
    elif value == 'False':
        value = False
    elif is_convertible_to_float(value):
        value = float(value)
    elif value.isdigit():
        value = int(value)
    elif value.startswith('[') and value.endswith(']'):
        value = value[1:-1]
        value = value.split(',')
        value = list(filter(None, value))
    elif ',' in value:
        value = value.split(',')
        value = list(filter(None, value))
    else:
        value = str(value)
    keys_split = key_list.replace('-', '_').split(':')
    return_dict = {keys_split[-1]: value}

    for key in reversed(keys_split[:-1]):
        return_dict = {key.replace('-', '_'): return_dict}
    return return_dict


def seed_everything(seed: int) -> None:
    """Set global random seed for reproducibility."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def split_prompt_response(
    texts: list[str],
    split_token: str,
) -> tuple[list[str], list[str]]:
    """Split prompt-response pairs into prompts and responses."""

    def split_fn(text: str) -> tuple[str, str]:
        """Split a prompt-response pair into prompt and response."""
        prompt, partition, response = text.rpartition(split_token)
        assert prompt and partition and response, f'invalid text: {text}'
        return prompt + partition, response

    return tuple(map(list, zip(*map(split_fn, texts))))


def gather_log_probabilities(
    logits: torch.Tensor,  # size = (B, L, V)
    labels: torch.LongTensor,  # size = (B, L)
) -> torch.Tensor:  # size = (B, L)
    """Gather log probabilities of the given labels from the logits."""
    log_probs = F.log_softmax(logits, dim=-1)  # size = (B, L, V)
    gathered_log_probs = torch.gather(  # size = (B, L, 1)
        log_probs,
        dim=-1,
        index=labels.unsqueeze(dim=-1),
    )
    return gathered_log_probs.squeeze(dim=-1)  # size = (B, L)


def batch_retokenize(
    input_ids: torch.LongTensor,
    src_tokenizer: PreTrainedTokenizerBase,
    dest_tokenizer: PreTrainedTokenizerBase,
    *,
    padding: bool | str | PaddingStrategy = PaddingStrategy.LONGEST,
    truncation: bool | str | TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
    skip_special_tokens: bool = True,
    device: torch.device | str | int | None = None,
) -> BatchEncoding:
    """Re-tokenize a batch of input ids from one tokenizer to another."""
    return dest_tokenizer(
        [
            text + dest_tokenizer.eos_token
            for text in src_tokenizer.batch_decode(
                input_ids.to(device),
                skip_special_tokens=skip_special_tokens,
            )
        ],
        padding=padding,
        truncation=truncation,
        return_tensors='pt',
    )


def is_same_tokenizer(
    tokenizer: PreTrainedTokenizerBase,
    other_tokenizer: PreTrainedTokenizerBase,
) -> bool:
    """Check if two tokenizers are the same."""
    return tokenizer is other_tokenizer or (
        tokenizer.__class__ == other_tokenizer.__class__
        and tokenizer.get_vocab() == other_tokenizer.get_vocab()
    )


def is_same_processor(
    processor: ProcessorMixin,
    other_processor: ProcessorMixin,
) -> bool:
    """Check if two processors are the same."""
    return processor is other_processor or (processor.__class__ == other_processor.__class__)


def masked_mean(
    x: torch.Tensor,  # size = (B, L)
    mask: torch.BoolTensor | None = None,  # size = (B, L)
) -> torch.Tensor:  # size = ()
    """Compute the mean of a tensor with a mask."""
    if mask is None:
        return x.mean()
    return ((x * mask).sum(dim=-1) / mask.sum(dim=-1)).mean()


def str2bool(string: str) -> bool:
    """Convert a string literal to a boolean value."""
    if string.lower() in {'1', 'true', 't', 'yes', 'y', 'on'}:
        return True
    if string.lower() in {'0', 'false', 'f', 'no', 'n', 'off'}:
        return False
    return bool(string)


def parse_unknown_args():
    parser = argparse.ArgumentParser(description='Parse bash arguments.')

    # parse unknown arguments
    _, unknown = parser.parse_known_args()
    args_dict = {}
    key = None
    for arg in unknown:
        if arg.startswith('--'):
            if key:
                args_dict[key] = True
            key = arg.lstrip('--')
        else:
            if key:
                args_dict[key] = arg
                key = None
    if key:
        args_dict[key] = True

    return args_dict
