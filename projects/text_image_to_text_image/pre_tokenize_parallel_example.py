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
import uuid

from align_anything.utils.template_registry import register_template


import torch
import torch.multiprocessing as mp

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

def format_sample_cham(raw_sample: dict[str, Any]) -> dict[str, Any]:
    """ Formating input sample, and change the related keys according to the training dataset."""
    """If you are using a different dataset, you need to customize this function or write a new function."""
    system_prompt: str = 'BEGINNING OF CONVERSATION: '
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
    elif input_img is None:
        input_images = []
        num_imput_img = 0
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
    elif output_img is None:
        output_images = []
        num_output_img = 0
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
    
    if len(input_images)==0 and len(output_images)==0 :
        return {
            'text': text,
            'prompt': prompt,
            'input_image': None,
            'image': None,
        }
    elif len(input_images)==0:
        return {
            'text': text,
            'prompt': prompt,
            'input_image': None,
            'image': output_images,
        }
    else:
        return {
            'text': text,
            'prompt': prompt,
            'input_image': input_images,
            'image': input_images + output_images,
        }
    
def format_sample(raw_sample: dict[str, Any]) -> dict[str, Any]:
    return format_sample_cham(raw_sample)


def preprocess(tokenizer, processor, formatted_sample: dict[str, Any]):
    return_dict = {}
    raw_text = ''
    if isinstance(formatted_sample['text'], list):
        raw_text = tokenizer.eos_token.join(formatted_sample['text'])
    elif isinstance(formatted_sample['text'], str):
        raw_text = formatted_sample['text'] + tokenizer.eos_token
    else:
        raise NotImplementedError
        
    if formatted_sample['image'] is None:
        text_dict = processor(raw_text, return_tensors='pt').to(dtype = torch.bfloat16)
        return_dict['input_ids'] = text_dict['input_ids'].squeeze()
        formatted_prompt = formatted_sample['prompt']
        prompt_dict = processor(formatted_prompt, return_tensors='pt').to(dtype = torch.bfloat16)
        labels = return_dict['input_ids'].clone()
        
        labels[: len(prompt_dict['input_ids'][0] - 1)] = IGNORE_INDEX
        return_dict['labels'] = labels
        return_dict['pixel_values'] = None
        return return_dict, len(prompt_dict['input_ids'][0] - 1), False
    else: 
        text_dict = processor(raw_text, formatted_sample['image'], return_tensors='pt').to(dtype = torch.bfloat16)
        
        return_dict['input_ids'] = text_dict['input_ids'].squeeze()

        formatted_prompt = formatted_sample['prompt']
        if formatted_sample['input_image'] is None:
            prompt_dict = processor(formatted_prompt, return_tensors='pt').to(dtype = torch.bfloat16)
        else:
            prompt_dict = processor(formatted_prompt, formatted_sample['input_image'], return_tensors='pt').to(dtype = torch.bfloat16)
        
        labels = return_dict['input_ids'].clone()
        # mask non-assistant input
        labels[: len(prompt_dict['input_ids'][0]) - 1] = IGNORE_INDEX
        return_dict['labels'] = labels

        return_dict['pixel_values'] = text_dict['pixel_values']
        
        return return_dict, len(prompt_dict['input_ids'][0]) - 1, True


def process_data(gpu, input_data, model_path, output_path, cache_dir):
    device = f"cuda:{gpu}"
    model = AccustomedChameleonModel.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map=device)
    processor = ChameleonProcessor.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    local_output_paths = []
    max_length = 0
    num_img = 0
    num_text = 0
    for piece in tqdm(input_data, desc=f"Processing on GPU {gpu}"):
        formatted_sample = format_sample(piece)
        preprocessed_sample, label_len, flag = preprocess(tokenizer, processor, formatted_sample)
        if flag:
            num_img += 1
        else:
            num_text += 1
        
        if preprocessed_sample['pixel_values'] is not None:
            updated_piece = model.pre_tokenization(
                input_ids=preprocessed_sample['input_ids'], 
                pixel_values=preprocessed_sample['pixel_values'].to(device = model.device, dtype = torch.bfloat16),
            )
        else:
            updated_piece = model.pre_tokenization(
                input_ids=preprocessed_sample['input_ids']
            )
        updated_piece['labels'] = updated_piece['input_ids'].clone()
        updated_piece['labels'][:label_len] = IGNORE_INDEX
        if updated_piece['input_ids'].shape[0] <= 4096:
            for key, value in updated_piece.items():
                if torch.is_tensor(value):
                    updated_piece[key] = value.cpu()
            file_name = str(uuid.uuid4()) + '.pt'
            file_path = os.path.join(cache_dir, file_name)
            torch.save(updated_piece, file_path)
            local_output_paths.append(file_path)
            
        # Clean up memory
        del updated_piece
        torch.cuda.empty_cache()

    output_path.extend(local_output_paths)
    print(f"GPU {gpu} processed {len(local_output_paths)} messages")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, default=".cache")
    parser.add_argument("--num_processes", type=int, default=8)
    parser.add_argument("--num_gpus", type=int, default=8)
    
    args = parser.parse_args()

    input_path = args.input_path
    output_path = args.output_path
    model_path = args.model_path
    cache_path = args.cache_dir
    
    # if cache dir does not exist, make one
    if not os.path.exists(cache_path):
        os.makedirs(cache_path)
    
    num_processes = args.num_processes
    num_gpus = args.num_gpus
    mp.set_start_method('spawn', force=True)
    output_paths = mp.Manager().list()  # For collecting results from multiple processes
    
    # change this logic into load_dataset if needed
    with open(input_path, 'r') as f:
        input_data = json.load(f)
    
    target = input_data # add to_list() if you acquire the dataset from load_dataset
    chunks = [target[i::num_processes] for i in range(num_processes)]
        
    processes = []
    for id in range(num_processes):
        gpu = id % num_gpus  # This maps process to GPU cyclically
        p = mp.Process(target=process_data, args=(gpu, chunks[id], model_path, output_paths, ".cache"))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
    
    output_paths = list(output_paths)
    
    all_data = []
    for path in output_paths:
        data = torch.load(path)
        all_data.append(data) 
        
    torch.set_printoptions(threshold=torch.inf)
    print(f"Effective Length: {len(all_data)}")
    
    torch.save(all_data, output_path)
        

if __name__ == "__main__":
    main()