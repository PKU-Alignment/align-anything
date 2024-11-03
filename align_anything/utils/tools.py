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
import cv2
import glob
import math
import random
import base64
from collections import namedtuple
from typing import Any, NamedTuple
import yt_dlp
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import yaml
from PIL import Image
import torch.utils.data
from scipy.stats import entropy
from moviepy.editor import AudioFileClip
from torch.nn.utils.rnn import pad_sequence
from torch.types import Number
from torch.autograd import Variable
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.models.inception import inception_v3
from transformers import PreTrainedTokenizerBase, ProcessorMixin
from transformers.tokenization_utils import BatchEncoding, PaddingStrategy, TruncationStrategy
import pickle


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
            keys = key[len(env_prefix):].lower().split('__')
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
    seed = int(seed)
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

def remove_pad_tokens(response: list[int], pad_token_id: int) -> list[int]:
    return [token for token in response if token != pad_token_id]

def download_video(url, video_path):
    ydl_opts = {
        'format': 'best',
        'outtmpl': video_path,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        return True
    except yt_dlp.utils.DownloadError as e:
        print(f"Error downloading {url}: {e}")
        return False
    
def save_raw_outputs(raw_outputs, raw_outputs_dir):
    with open(raw_outputs_dir, 'wb') as f:
        pickle.dump(raw_outputs, f)

def load_raw_outputs(raw_outputs_dir):
    with open(raw_outputs_dir, 'rb') as f:
        inference_output = pickle.load(f)
    return inference_output

def download_audio(youtube_id, start_time, audiocap_id, output_dir):
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': os.path.join(output_dir, f'{audiocap_id}.webm'),
        'noplaylist': True,
    }
    
    youtube_url = f'https://www.youtube.com/watch?v={youtube_id}'
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_url])
    except yt_dlp.utils.DownloadError as e:
        print(f"Download failed for {youtube_url}. Error: {e}")
        return None, f"Download failed for {youtube_url}. Error: {e}"
    
    audio_path = os.path.join(output_dir, f'{audiocap_id}.webm')
    
    try:
        audio_clip = AudioFileClip(audio_path)
        end_time = audio_clip.duration
        start_time_sec = float(start_time)
        audio_segment = audio_clip.subclip(start_time_sec, min(end_time, start_time_sec + 10))
        
        output_audio_path = os.path.join(output_dir, f'{audiocap_id}.mp3')
        audio_segment.write_audiofile(output_audio_path)
        
        os.remove(audio_path)
        if output_audio_path.endswith('.mp3') and os.path.isfile(output_audio_path):
            return output_audio_path, ""
        return None, ".webm file connot be saved."
    except Exception as e:
        print(f"Error processing audio for {audiocap_id}. Error: {e}")
        return None, f"Error processing audio for {audiocap_id}. Error: {e}"

def image_crop(input_folder):
    output_folder = f'{input_folder}_crop'
    os.makedirs(output_folder, exist_ok=True)
    for filename in os.listdir(input_folder):
        img_path = os.path.join(input_folder, filename)
        if os.path.isfile(img_path):
            try:
                with Image.open(img_path) as img:
                    img = img.convert("RGB")
                    img_resized = img.resize((1024, 1024))
                    output_path = os.path.join(output_folder, filename)
                    img_resized.save(output_path)
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                
    return output_folder

def inception_score(imgs, cuda=True, batch_size=32, resize=False, splits=1):
    N = len(imgs)
    assert batch_size > 0
    assert N > batch_size

    device = torch.device("cuda" if cuda and torch.cuda.is_available() else "cpu")

    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)
    inception_model = inception_v3(pretrained=True, transform_input=False).to(device)
    inception_model.eval()
    up = nn.Upsample(size=(299, 299), mode='bilinear').to(device)
    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x).data.cpu().numpy()

    preds = np.zeros((N, 1000))

    for i, batch in enumerate(dataloader, 0):
        batch = batch.to(device)
        batchv = Variable(batch)
        batch_size_i = batch.size()[0]

        preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batchv)

    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)

def resize_frame(frame, short_edge=256):
    height, width = frame.shape[:2]
    if min(height, width) <= short_edge:
        return frame
    else:
        scale = short_edge / width if height > width else short_edge / height
        new_width = int(width * scale)
        new_height = int(height * scale)
        resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
        return resized_frame

def get_frames(video_path, output_folder, num_frames=8):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_to_capture = set([0, total_frames - 1])
    frames_interval = (total_frames - 1) // (num_frames - 1)
    for i in range(1, num_frames - 1):
        frames_to_capture.add(i * frames_interval)

    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count in frames_to_capture:
            resized_frame = resize_frame(frame)
            frame_name = f"{os.path.splitext(os.path.basename(video_path))[0]}_frame{count}.png"
            output_path = os.path.join(output_folder, frame_name)
            cv2.imwrite(output_path, resized_frame)
        count += 1

    cap.release()

def process_videos(video_id, video_path, frames_dir):
    video_images = glob.glob(os.path.join(frames_dir, f"{video_id}_frame*.png"))
    
    if len(video_images) == 8:
        return
    for img in video_images:
        os.remove(img)

    get_frames(video_path, frames_dir)
    frames = sorted(glob.glob(os.path.join(frames_dir, f"{video_id}_frame*.png")))
    return [os.path.basename(frame) for frame in frames]

def image_b64(image_path):
    with open(image_path, "rb") as image_file:
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
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])

def smart_resize(height: int, width: int, factor: int = 28, min_pixels: int = 4 * 28 * 28, max_pixels: int = 16384 * 28 * 28):
    if max(height, width) / min(height, width) > 200:
        raise ValueError(f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}")
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
    fps = ele.get("fps", 2.0)
    min_frames = math.ceil(ele.get("min_frames", 4) / 2) * 2
    max_frames = math.floor(ele.get("max_frames", min(768, total_frames)) / 2) * 2
    nframes = total_frames / video_fps * fps
    nframes = min(max(nframes, min_frames), max_frames)
    nframes = round(nframes / 2) * 2
    if not (2 <= nframes and nframes <= total_frames):
        raise ValueError(f"nframes should in interval [2, {total_frames}], but got {nframes}.")
    return nframes

def read_video(ele: dict, task_idx):
    import decord
    video_path = ele["video"]
    vr = decord.VideoReader(video_path)
    total_frames, video_fps = len(vr), vr.get_avg_fps()
    nframes = smart_nframes(ele, total_frames=total_frames, video_fps=video_fps)
    nframes = min(task_idx, nframes)
    idx = torch.linspace(0, total_frames - 1, nframes).round().long().tolist()
    video = vr.get_batch(idx).asnumpy()
    video = torch.tensor(video).permute(0, 3, 1, 2)
    return video

def fetch_video(ele: dict, task_idx, image_factor: int = 28):
    video = read_video(ele, task_idx)
    nframes, _, height, width = video.shape

    min_pixels = ele.get("min_pixels", 128 * 28 * 28)
    total_pixels = ele.get("total_pixels", 24576 * 28 * 28)
    max_pixels = max(min(768 * 28 * 28, total_pixels / nframes * 2), int(min_pixels * 1.05))
    max_pixels = ele.get("max_pixels", max_pixels)
    if "resized_height" in ele and "resized_width" in ele:
        resized_height, resized_width = smart_resize(ele["resized_height"], ele["resized_width"], factor=image_factor)
    else:
        resized_height, resized_width = smart_resize(height, width, factor=image_factor, min_pixels=min_pixels, max_pixels=max_pixels)
    video = transforms.functional.resize(video, [resized_height, resized_width], interpolation=InterpolationMode.BICUBIC, antialias=True).float()
    return video

def extract_vision_info(conversations: list[dict] | list[list[dict]]):
    vision_infos = []
    if isinstance(conversations[0], dict):
        conversations = [conversations]
    for conversation in conversations:
        for message in conversation:
            if isinstance(message["content"], list):
                for ele in message["content"]:
                    if ("video" in ele or ele["type"] in ("video")):
                        vision_infos.append(ele)
    return vision_infos

def process_vision(conversations: list[dict] | list[list[dict]], task_idx):
    vision_infos = extract_vision_info(conversations)
    video_inputs = []
    for vision_info in vision_infos:
        if "video" in vision_info:
            video_inputs.append(fetch_video(vision_info, task_idx))
        else:
            raise ValueError("video should in content.")
    return video_inputs

def count_right_padding(lst, padding=0):
    """Counts the number of padding values (default is 0) on the right side of a list.

    This function iterates over the elements of the given list from the end to the start.
    It stops counting when it encounters the first non-padding element.

    Args:
        lst (List): The list to be checked.
        padding (int, optional): The value considered as padding. Defaults to 0.

    Returns:
        int: The number of padding values on the right side of the list.
    """
    count = 0
    # Iterate over the list in reverse order
    for i in range(len(lst) - 1, -1, -1):
        if lst[i] == padding:
            count += 1
        else:
            # Stop counting when a non-padding value is encountered
            break

    return count