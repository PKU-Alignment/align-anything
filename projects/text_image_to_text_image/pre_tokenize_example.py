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



import io
import os
import json
from abc import ABC, abstractmethod
import copy
from typing import Any, Callable
from typing_extensions import TypedDict  # Python 3.10+
from tqdm import tqdm

import requests
import librosa
from PIL import Image
from torchvision.io import read_video

from align_anything.utils.template_registry import register_template


import torch

import transformers
from torch.utils.data import Dataset
from torchvision import transforms
from transformers.tokenization_utils import PaddingStrategy, TruncationStrategy
from transformers import AutoTokenizer, ChameleonProcessor
from align_anything.models.chameleon_model import AccustomedChameleonModel

from align_anything.utils.multi_process import get_current_device
from align_anything.utils.template_registry import get_template_class
from align_anything.utils.tools import right_padding
from datasets import load_dataset
import argparse

ALLOWED_ATTRIBUTES = ['split_token']
DEFAULT_SPLIT_TOKEN = 'ASSISTANT:'
IGNORE_INDEX = -100


def load_image(image_path: str):
    try:
        if image_path.startswith("http"):
            image = Image.open(requests.get(image_path, stream=True).raw)
        else:
            image = Image.open(image_path)
        return image
    except Exception as e:
        print(f"Error occured when dealing with {image_path}")
        raise Exception


def format_sample(raw_sample: dict[str, Any]) -> dict[str, Any]:
    system_prompt: str = ''
    user_prompt: str = 'USER: \n{input}'
    assistant_prompt: str = '\nASSISTANT:{output}'
    split_token: str = 'ASSISTANT:'
    separator: str = '###'
    input_text = raw_sample['question']
    output_text = raw_sample['response']
    input_img = raw_sample['image_url']
    output_img = raw_sample['output_image_url']
    
    if isinstance(input_img, str):
        input_images = [load_image(input_img)]
        num_imput_img = 1
    elif isinstance(input_img, list):
        input_images = [load_image(img) for img in input_img]
        num_input_img = len(input_img)
    else:
        raise ValueError("input_image must be either a string or a list of strings")
    
    input_text = f"{'<image>' * num_imput_img}{input_text}"
    
    # do the same for output
    if isinstance(output_img, str):
        output_images = [load_image(output_img)]
        num_output_img = 1
        
    elif isinstance(output_img, list):
        output_images = [load_image(img) for img in output_img]
        num_output_img = len(output_img)
    else:
        raise ValueError("output_image must be either a string or a list of strings")
    
    output_text = f"{'<image>' * num_output_img}{output_text}"
    
    text = (
        f'{system_prompt}'
        f'{user_prompt.format(input=input_text)}'
        f"{assistant_prompt.format(output=output_text)}"
    )

    prompt = (
        f'{system_prompt}'
        f'{user_prompt.format(input=input_text)}'
        f"{assistant_prompt.format(output='')}"
    )
    
    return {
        'text': text,
        'prompt': prompt,
        'input_image': input_images,
        'image': input_images + output_images,
    }
    

def preprocess(tokenizer, processor, formatted_sample: dict[str, Any]):
    return_dict = {}
    raw_text = ''
    if isinstance(formatted_sample['text'], list):
        raw_text = tokenizer.eos_token.join(formatted_sample['text'])
    elif isinstance(formatted_sample['text'], str):
        raw_text = formatted_sample['text'] + tokenizer.eos_token
    else:
        raise NotImplementedError

    text_dict = processor(raw_text, formatted_sample['image'], return_tensors='pt').to(dtype = torch.bfloat16)
    
    return_dict['input_ids'] = text_dict['input_ids'].squeeze()

    formatted_prompt = formatted_sample['prompt']
    prompt_dict = processor(formatted_prompt, formatted_sample['input_image'], return_tensors='pt').to(dtype = torch.bfloat16)
    
    labels = return_dict['input_ids'].clone()
    # mask non-assistant input
    labels[: len(prompt_dict['input_ids'])] = IGNORE_INDEX
    return_dict['labels'] = labels

    return_dict['pixel_values'] = text_dict['pixel_values']
    
    return return_dict, len(prompt_dict['input_ids'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    
    args = parser.parse_args()

    input_path = args.input_path
    output_path = args.output_path
    model_path = args.model_path
    
    model = AccustomedChameleonModel.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="cuda")
    processor = ChameleonProcessor.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    with open(input_path, 'r') as f:
        input_data = json.load(f)
        
    output_data = []
    max_length = 0 
    for piece in tqdm(input_data, desc="Processing data"):
        formatted_sample = format_sample(piece)
        preprocessed_sample, label_len = preprocess(tokenizer, processor, formatted_sample)
        
        updated_piece = model.pre_tokenization(
            input_ids=preprocessed_sample['input_ids'], 
            pixel_values=preprocessed_sample['pixel_values'].to(device = model.device, dtype = torch.bfloat16),
        )
        updated_piece['labels'] = updated_piece['input_ids'].clone()
        updated_piece['labels'][:label_len] = IGNORE_INDEX
        
        if updated_piece['input_ids'].shape[0] <= 4096:
            output_data.append(updated_piece)
    
    print(f"Effective Length: {len(output_data)}")
    torch.save(output_data, output_path)
        


if __name__ == "__main__":
    main()