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
import base64
import json
import math
import os
import pickle
import random
from collections import namedtuple
from typing import Any, NamedTuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import yaml
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.types import Number
from transformers import PreTrainedTokenizerBase, ProcessorMixin
from transformers.image_utils import ImageInput
from transformers.tokenization_utils import BatchEncoding, PaddingStrategy, TruncationStrategy
from transformers.utils.import_utils import requires_backends

from align_anything.utils.device_utils import manual_seed_all


def convert_to_rgb(image: ImageInput) -> ImageInput:
    """
    Converts an image to RGB format. Only converts if the image is of type PIL.Image.Image, otherwise returns the image
    as is.
    Args:
        image (Image):
            The image to convert.
    """
    requires_backends(convert_to_rgb, ['vision'])

    if not isinstance(image, Image.Image):
        return image

    if image.mode == 'RGB':
        return image

    image = image.convert('RGB')
    image = np.array(image)
    return image


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
    if obj is None:
        return {}
    if isinstance(obj, tuple) and hasattr(obj, '_fields'):
        return {field: namedtuple_to_dict(getattr(obj, field)) for field in obj._fields}
    elif isinstance(obj, list):
        return [namedtuple_to_dict(item) for item in obj]
    else:
        return obj


def requestoutput_to_dict(data, mode='brief'):
    if mode == 'brief':
        info = {'prompt': data.prompt, 'outputs': []}
    else:
        info = {
            'prompt': data.prompt,
            'prompt_token_ids': data.prompt_token_ids,
            'prompt_logprobs': [
                vllm_logprob_to_dict(token_logprob) for token_logprob in data.prompt_logprobs[1:]
            ],
            'outputs': [],
            'finished': data.finished,
            'metrics': {
                'arrival_time': data.metrics.arrival_time,
                'last_token_time': data.metrics.last_token_time,
                'first_scheduled_time': data.metrics.first_scheduled_time,
                'first_token_time': data.metrics.first_token_time,
                'time_in_queue': data.metrics.time_in_queue,
                'finished_time': data.metrics.finished_time,
            },
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
                'logprobs': [
                    vllm_logprob_to_dict(token_logprob) for token_logprob in output.logprobs
                ],
                'finish_reason': output.finish_reason,
                'stop_reason': output.stop_reason,
            }
        info['outputs'].append(output)
        return info


def vllm_logprob_to_dict(data):
    return [{v.decoded_token: v.logprob} for k, v in data.items()]


def set_nested_value(dictionary, keys, value):
    for key in keys[:-1]:
        dictionary = dictionary.setdefault(key, {})
    dictionary[keys[-1]] = value


def override_nested_value(config, keys, value):
    for key, subconfig in config.items():
        if isinstance(subconfig, dict):
            override_nested_value(subconfig, keys, value)
    if keys[0] in config:
        set_nested_value(config, keys, value)


def override_with_env_variables(config, env_prefix):
    for key, value in os.environ.items():
        if key.startswith(env_prefix):
            keys = key[len(env_prefix) :].lower().split('__')
            override_nested_value(config, keys, value)


def yaml_load(yaml_path):

    # Use the PREFIX ENV PREFIX to identify the relevant environment variables
    env_prefix = 'ENV_PREFIX__'
    with open(yaml_path, encoding='utf-8') as f:
        try:
            configs = yaml.safe_load(f)
            override_with_env_variables(configs, env_prefix)
            return configs
        except FileNotFoundError as exc:
            raise FileNotFoundError(f'{yaml_path} error: {exc}') from exc


def read_cfgs(mode: str, task: str) -> list[dict[str, Any], dict[str, Any]]:
    current_file_path = os.path.abspath(__file__)
    parent_path = os.path.dirname(os.path.dirname(current_file_path))
    yaml_path = os.path.join(parent_path, 'configs', mode, f'{task}.yaml')

    configs = yaml_load(yaml_path)
    zero_stage_file = os.getenv('ZERO_STAGE_FILE', configs['train_cfgs']['ds_cfgs'])

    ds_cfgs_path = os.path.join(
        parent_path,
        'configs',
        'deepspeed',
        zero_stage_file,
    )
    with open(ds_cfgs_path) as f:
        ds_cfgs = json.load(f)

    os.environ['ZERO_STAGE'] = str(ds_cfgs['zero_optimization']['stage'])
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
    no_decay_name_list=[
        'bias',
        'layer_norm.weight',
        'layernorm.weight',
        'norm.weight',
        'ln_f.weight',
    ],
) -> list[dict[str, list[nn.Parameter] | float]]:
    """Get parameter groups with customized weight decay value."""
    return [
        {
            'params': [
                p
                for n, p in module.named_parameters()
                if (not any(nd in n for nd in no_decay_name_list) and p.requires_grad)
            ],
            'weight_decay': weight_decay,
        },
        {
            'params': [
                p
                for n, p in module.named_parameters()
                if (any(nd in n for nd in no_decay_name_list) and p.requires_grad)
            ],
            'weight_decay': 0.0,
        },
    ]


def prepare_ds_train_cfgs(custom_cfgs: NamedTuple, raw_ds_cfgs: dict[str, Any]) -> dict[str, Any]:
    """Prepare the DeepSpeed config for training."""
    ds_cfgs = raw_ds_cfgs.copy()
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    micro_batch_size_per_gpu = int(custom_cfgs.per_device_train_batch_size)
    gradient_accumulation_steps = int(custom_cfgs.gradient_accumulation_steps)

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
    elif value.isdigit():
        value = int(value)
    elif is_convertible_to_float(value):
        value = float(value)
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
    seed = int(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    manual_seed_all(seed)


def split_prompt_response(
    texts: list[str],
    split_token: str,
) -> tuple[list[str], list[str]]:
    """Split prompt-response pairs into prompts and responses."""

    def split_fn(text: str) -> tuple[str, str]:
        """Split a prompt-response pair into prompt and response."""
        prompt, response = text.split(split_token, maxsplit=1)
        assert prompt and response, f'invalid text: {text}'
        return prompt, response

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
        index=labels.unsqueeze(dim=-1).to(torch.int64),
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


def remove_pad_tokens(response: list[int], pad_token_id: int) -> list[int]:
    return [token for token in response if token != pad_token_id]


def save_raw_outputs(raw_outputs, raw_outputs_dir):
    with open(raw_outputs_dir, 'wb') as f:
        pickle.dump(raw_outputs, f)


def load_raw_outputs(raw_outputs_dir):
    with open(raw_outputs_dir, 'rb') as f:
        inference_output = pickle.load(f)
    return inference_output


def image_crop(input_folder):
    output_folder = f'{input_folder}_crop'
    os.makedirs(output_folder, exist_ok=True)
    for filename in os.listdir(input_folder):
        img_path = os.path.join(input_folder, filename)
        if os.path.isfile(img_path):
            try:
                with Image.open(img_path) as img:
                    img = img.convert('RGB')
                    img_resized = img.resize((1024, 1024))
                    output_path = os.path.join(output_folder, filename)
                    img_resized.save(output_path)
            except Exception as e:
                print(f'Error processing {filename}: {e}')

    return output_folder


def image_b64(image_path):
    with open(image_path, 'rb') as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def read_video_pyav(container, indices):
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format='rgb24') for x in frames])


def smart_resize(
    height: int,
    width: int,
    factor: int = 28,
    min_pixels: int = 4 * 28 * 28,
    max_pixels: int = 16384 * 28 * 28,
):
    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            f'absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}'
        )
    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def smart_nframes(ele: dict, total_frames: int, video_fps: int | float):
    fps = ele.get('fps', 2.0)
    min_frames = math.ceil(ele.get('min_frames', 4) / 2) * 2
    max_frames = math.floor(ele.get('max_frames', min(768, total_frames)) / 2) * 2
    nframes = total_frames / video_fps * fps
    nframes = min(max(nframes, min_frames), max_frames)
    nframes = round(nframes / 2) * 2
    if not (2 <= nframes and nframes <= total_frames):
        raise ValueError(f'nframes should in interval [2, {total_frames}], but got {nframes}.')
    return nframes


def extract_vision_info(conversations: list[dict] | list[list[dict]]):
    vision_infos = []
    if isinstance(conversations[0], dict):
        conversations = [conversations]
    for conversation in conversations:
        for message in conversation:
            if isinstance(message['content'], list):
                for ele in message['content']:
                    if 'video' in ele or ele['type'] in ('video'):
                        vision_infos.append(ele)
    return vision_infos


def ends_with_any(s, substrings):
    """Check if the string ends with any of the substrings.

    Args:
        s (str): The string to check.
        substrings (list[str]): The list of substrings to check.

    Returns:
        bool: True if the string ends with any of the substrings, False otherwise.
    """
    temp = s.strip()
    return temp.endswith(tuple(substrings))


def move_padding_left(input_tensor, padding_value=0):
    """Moves the padding values in each row of the input_tensor from the right to the left.

    Args:
        input_tensor (Tensor): A 2D tensor to be processed.
        padding_value (int): The value used for padding, default is 0.

    Returns:
        Tensor: The tensor with padding values moved to the left.
    """
    start_pad_counts = (
        (input_tensor == padding_value)
        .cumsum(dim=1)
        .eq(torch.arange(1, input_tensor.size(1) + 1, device=input_tensor.device))
        .sum(dim=1)
    )
    non_pad_counts = (input_tensor != padding_value).sum(dim=1)
    output_tensor = torch.full_like(input_tensor, padding_value, device=input_tensor.device)
    max_len = input_tensor.size(1)
    indices = torch.arange(max_len, device=input_tensor.device).expand(len(non_pad_counts), max_len)
    shifts = max_len - non_pad_counts.unsqueeze(1) - start_pad_counts.unsqueeze(1)
    new_indices = (indices - shifts) % max_len
    output_tensor = torch.gather(input_tensor, 1, new_indices)

    return output_tensor


def strip_pad(seq: torch.Tensor, pad_token_id: int):
    return seq[seq != pad_token_id]
