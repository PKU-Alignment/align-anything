# Copyright 2025 PKU-Alignment Team. All Rights Reserved.
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
import argparse
import json
import os
import uuid

import requests
import torch
import torch.multiprocessing as mp
from janus.models import MultiModalityCausalLM, VLChatProcessor, VLMImageProcessor
from PIL import Image
from tqdm import tqdm

from align_anything.utils.device_utils import set_device, torch_gc


ignore_index = -100


def load_image(image_path: str):
    try:
        if image_path.startswith('http'):
            image = Image.open(requests.get(image_path, stream=True).raw).convert('RGB')
        else:
            image = Image.open(image_path).convert('RGB')
        return image
    except Exception as e:
        print(f'Error occurred when dealing with {image_path}: {e}')
        raise Exception


def format_sample_janus(piece, vl_chat_processor):
    sample = {
        'input_text': piece['prompt'],
        'output_image': load_image(piece['image']),
    }
    return sample


def tokenize_sample(vl_chat_processor, vl_gpt, vl_image_processor, formatted_sample):
    conversation = [
        {'role': 'User', 'content': formatted_sample['input_text']},
        {'role': 'Assistant', 'content': ''},
    ]
    sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
        conversations=conversation,
        sft_format=vl_chat_processor.sft_format,
        system_prompt='',
    )

    prompt = sft_format + vl_chat_processor.image_start_tag
    input_ids = vl_chat_processor.tokenizer.encode(prompt)
    input_ids = torch.LongTensor(input_ids).to(vl_gpt.device)

    pixel_values = (
        vl_image_processor([formatted_sample['output_image']], return_tensors='pt')['pixel_values']
        .to(vl_gpt.device)
        .to(torch.bfloat16)
    )
    (
        quant,
        (vq_loss, commit_loss, entropy_loss),
        (perplexity, min_encodings, min_encoding_indices),
    ) = vl_gpt.gen_vision_model.encode(pixel_values)
    full_input_ids = torch.cat([input_ids, min_encoding_indices])
    labels = full_input_ids.clone()
    labels[: len(input_ids)] = ignore_index

    return {
        'input_ids': full_input_ids.to('cpu'),
        'labels': labels.to('cpu'),
        'task': 'generation',
    }


def process_data(gpu, chunk, model_path, output_paths, cache_path):
    device = set_device(gpu)
    print(f'Initializing Model on {device}')
    vl_chat_processor = VLChatProcessor.from_pretrained(model_path, device=device)
    vl_gpt = MultiModalityCausalLM.from_pretrained(model_path, trust_remote_code=True).to(device)
    vl_gpt = vl_gpt.to(torch.bfloat16).eval()
    vl_image_processor = VLMImageProcessor.from_pretrained(model_path, device=device)

    print(f'Finished Initializing Model on {device}')

    local_output_paths = []
    for piece in tqdm(chunk, desc=f'Processing on GPU {gpu}'):
        formatted_sample = format_sample_janus(piece, vl_chat_processor)
        sample = tokenize_sample(vl_chat_processor, vl_gpt, vl_image_processor, formatted_sample)
        file_name = str(uuid.uuid4()) + '.pt'
        file_path = os.path.join(cache_path, file_name)
        torch.save(sample, file_path)
        local_output_paths.append(file_path)
        del sample
        torch_gc()

    output_paths.extend(local_output_paths)
    print(f'Processed {len(local_output_paths)} samples on GPU {gpu}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--cache_dir', type=str, default='.cache')
    parser.add_argument('--num_processes', type=int, default=8)
    parser.add_argument('--num_gpus', type=int, default=8)

    args = parser.parse_args()

    input_path = args.input_path
    output_path = args.output_path
    model_path = args.model_path
    cache_path = args.cache_dir

    # if cache dir does not exist, make one
    if not os.path.exists(cache_path):
        os.makedirs(cache_path)

    with open(input_path) as f:
        input_data = json.load(f)

    num_processes = args.num_processes
    num_gpus = args.num_gpus
    mp.set_start_method('spawn', force=True)
    output_paths = mp.Manager().list()  # For collecting results from multiple processes

    target = input_data  # add to_list() if you acquire the dataset from load_dataset
    print(f'Full Length: {len(target)}')
    chunks = [target[i::num_processes] for i in range(num_processes)]

    processes = []
    for id in range(num_processes):
        gpu = id % num_gpus  # This maps process to GPU cyclically
        p = mp.Process(
            target=process_data, args=(gpu, chunks[id], model_path, output_paths, '.cache')
        )
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
    print(f'Effective Length: {len(all_data)}')

    torch.save(all_data, output_path)


if __name__ == '__main__':
    main()
