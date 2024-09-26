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

from curses import raw
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

def insert_img_token(text, image):
    if isinstance(image, str):
        decoded_images = [load_image(image)]
        num_images = 1
    elif isinstance(image, list):
        decoded_images = [load_image(img) for img in image]
        num_images = len(image)
    elif isinstance(image, Image.Image):
        decoded_images = [image]
        num_images = 1
    else:
        num_images = 0
        decoded_images = None
    
    processed_text = f"{'<image>' * num_images}{text}"
    return processed_text, decoded_images

def safe_add(list1, list2):
    if list1 is None and list2 is None:
        return []
    elif list1 is None:
        return list2.copy()
    elif list2 is None:
        return list1.copy()
    else:
        return list1 + list2
    
def format_sample_cham(raw_sample: dict[str, Any]) -> dict[str, Any]:
    """ Formating input sample, and change the related keys according to the training dataset."""
    """If you are using a different dataset, you need to customize this function or write a new function like the ones below."""
    system_prompt: str = ''
    user_prompt: str = 'USER: \n{input}'
    assistant_prompt: str = '\nASSISTANT:{output}'
    split_token: str = 'ASSISTANT:'
    separator: str = '###'

    input_text = raw_sample['input_text']
    input_img = raw_sample['input_image']
    
    input_text_processed, input_images = insert_img_token(input_text, input_img)
    
    text_full = (
        f'{system_prompt}'
        f'{user_prompt.format(input=input_text_processed)}'
        f"{assistant_prompt.format(output='')}"
    )
    
    return {
        'text': text_full,
        'images': input_images,
    }

def format_sample_pickapic(raw_sample: dict[str, Any]) -> dict[str, Any]:
    """Specified for pickapic dataset. You can customize a similar function for you own dataset."""
    prompt = f"Generate an image according to the following prompt: {raw_sample['caption']}"
    better_id = int(raw_sample['label_1'])
    worse_id = int(raw_sample['label_0'])

    raw_better_image = raw_sample[f'jpg_{better_id}']
    raw_worse_image = raw_sample[f'jpg_{worse_id}']
    better_image = Image.open(io.BytesIO(raw_better_image)).convert('RGB')
    worse_image = Image.open(io.BytesIO(raw_worse_image)).convert('RGB')
    raw_sample = {
        "input_text": prompt,
        "input_image": [],
    }
    return format_sample_cham(raw_sample)

def format_sample_spavl(raw_sample: dict[str, Any]) -> dict[str, Any]:
    """Specified for spavl dataset. You can customize a similar function for you own dataset."""
    raw_sample = {
        "input_text": raw_sample['question'],
        "input_image": load_image(raw_sample['image']),
    }
    return format_sample_cham(raw_sample)

def format_sample_AA(raw_sample: dict[str, Any]) -> dict[str, Any]:
    """Specified for align anything internal dataset. You can customize a similar function for you own dataset."""
    raw_sample = {
        "input_text": raw_sample['prompt'],
        "input_image": None,
    }
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
    
    if formatted_sample['images'] is not None:
        dict = processor(raw_text, formatted_sample['images'], return_tensors='pt').to(dtype = torch.bfloat16)
        return_dict['input_ids'] = dict['input_ids'].squeeze()
        return_dict['pixel_values'] = dict['pixel_values']
    else:
        dict = processor(raw_text, return_tensors='pt').to(dtype = torch.bfloat16)
        return_dict['input_ids'] = dict['input_ids'].squeeze()
        
    return return_dict


def process_data(gpu, input_data, model_path, output_path, cache_dir):
    device = f"cuda:{gpu}"
    print(f"Initializing Model on {device}")
    model = AccustomedChameleonModel.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map=device)
    processor = ChameleonProcessor.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print(f"Finished Initializing Model on {device}")
    local_output_paths = []
    for piece in tqdm(input_data, desc=f"Processing on GPU {gpu}"):
        formatted_sample = format_sample_AA(piece)
        preprocessed_sample = preprocess(tokenizer, processor, formatted_sample)
        
        
        if 'pixel_values' in preprocessed_sample.keys() and preprocessed_sample['pixel_values'] is not None:
            processed = model.pre_tokenization(
                input_ids=preprocessed_sample['input_ids'], 
                pixel_values=preprocessed_sample['pixel_values'].to(device = model.device, dtype = torch.bfloat16),
            )
        else:
            processed = model.pre_tokenization(
                input_ids=preprocessed_sample['input_ids']
            )

            
        updated_piece = {
            'input_ids': processed['input_ids'],
        }
        if updated_piece['input_ids'].shape[0] <= 4096:
            for key, value in updated_piece.items():
                if torch.is_tensor(value):
                    updated_piece[key] = value.cpu()
            file_name = str(uuid.uuid4()) + '.pt'
            file_path = os.path.join(cache_dir, file_name)
            torch.save(updated_piece, file_path)
            local_output_paths.append(file_path)
            
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
        
    with open(input_path, 'r') as f:
        input_data = json.load(f)
        
    num_processes = args.num_processes
    num_gpus = args.num_gpus
    mp.set_start_method('spawn', force=True)
    output_paths = mp.Manager().list()  # For collecting results from multiple processes
    
    target = input_data # add to_list() if you acquire the dataset from load_dataset
    print(f"Full Length: {len(target)}")
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