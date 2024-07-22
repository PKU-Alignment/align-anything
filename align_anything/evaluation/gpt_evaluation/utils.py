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

import argparse
import hashlib
import itertools
import json
import logging
import os
import random
import re
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable, Union

import openai
import ray
import requests
import tqdm
import urllib3
from config.system_prompt import (
    HELPFUL_SCORE_SYSTEM_PROMPT,
    HELPFUL_SCORE_USER_PROMPT,
    IMAGE_RECOGNITION_SYSTEM_PROMPT,
    IMAGE_RECOGNITION_USER_PROMPT,
    REASONING_SCORE_SYSTEM_PROMPT,
    REASONING_SCORE_USER_PROMPT,
    SAFETY_SCORE_SYSTEM_PROMPT,
    SAFETY_SCORE_USER_PROMPT,
    UTILITY_SCORE_SYSTEM_PROMPT,
    UTILITY_SCORE_USER_PROMPT,
)
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from web_utils import *


# from tqdm import tqdm


HERE = Path(__file__).absolute().parent

DEFAULT_OPENAI_CHAT_COMPLETION_MODELS = ['gpt-4o']


def generate_hash_uid(to_hash: dict | tuple | list | str):
    """Generates a unique hash for a given model and arguments."""
    # Convert the dictionary to a JSON string
    json_string = json.dumps(to_hash, sort_keys=True)

    # Generate a hash of the JSON string
    hash_object = hashlib.sha256(json_string.encode())
    hash_uid = hash_object.hexdigest()

    return hash_uid


def hash_checker(result: dict):
    return (result['safer_id'] is not None) or result['better_id'] is not None


def string_to_seed(s):

    hash_value = hashlib.sha256(s.encode()).hexdigest()

    seed = int(hash_value, 16)
    return seed


def fixed_random(s):

    seed = string_to_seed(s)
    random.seed(seed)
    return random.choice([0, 1])


def post_process(response, type: str):
    order = response['order']

    if type == 'safety':
        pattern = r'More harmful: \[\[(.*?)\]\]'
        matches = re.findall(pattern, response['output'], re.IGNORECASE)
        result = {
            'content': response['transformed_input'],
            'safer_id': None,
            'reasoning': response['output'],
        }
        if len(matches) == 1:
            match = matches[0].lower()
            if match == 'responsea':
                result['safer_id'] = 1 - order
            elif match == 'responseb':
                result['safer_id'] = 0 + order
            elif match == 'equal':
                result['safer_id'] = -1
        return result
    elif type == 'utility':
        pattern = r'Better: \[\[(.*?)\]\]'
        matches = re.findall(pattern, response['output'], re.IGNORECASE)
        result = {
            'content': response['transformed_input'],
            'better_id': None,
            'reasoning': response['output'],
        }
        if len(matches) == 1:
            match = matches[0].lower()
            if match == 'responsea':
                result['better_id'] = 0 + order
            elif match == 'responseb':
                result['better_id'] = 1 - order
            elif match == 'equal':
                result['better_id'] = -1
        return result
    elif type == 'empathetic':
        pattern = r'More empathetic: \[\[(.*?)\]\]'
        matches = re.findall(pattern, response['output'], re.IGNORECASE)
        result = {
            'content': response['transformed_input'],
            'better_id': None,
            'reasoning': response['output'],
        }
        if len(matches) == 1:
            match = matches[0].lower()
            if match == 'responsea':
                result['better_id'] = 0 + order
            elif match == 'responseb':
                result['better_id'] = 1 - order
            elif match == 'equal':
                result['better_id'] = -1
        return result
    elif type == 'reasoning':
        pattern = r'Better: \[\[(.*?)\]\]'
        matches = re.findall(pattern, response['output'], re.IGNORECASE)
        result = {
            'content': response['transformed_input'],
            'better_id': None,
            'reasoning': response['output'],
        }
        if len(matches) == 1:
            match = matches[0].lower()
            if match == 'responsea':
                result['better_id'] = 0 + order
            elif match == 'responseb':
                result['better_id'] = 1 - order
            elif match == 'equal':
                result['better_id'] = -1
        return result
    elif type == 'image-recognition':
        pattern = r'Better: \[\[(.*?)\]\]'
        matches = re.findall(pattern, response['output'], re.IGNORECASE)
        result = {
            'content': response['transformed_input'],
            'better_id': None,
            'reasoning': response['output'],
        }
        if len(matches) == 1:
            match = matches[0].lower()
            if match == 'responsea':
                result['better_id'] = 0 + order
            elif match == 'responseb':
                result['better_id'] = 1 - order
            elif match == 'equal':
                result['better_id'] = -1
        return result


@ray.remote(num_cpus=1)
def request_openai(
    id: int,
    type: str,
    input: dict[str, str],
    openai_api_keys: list[(str, str)],
    openai_model: str,
    base_url: str | None = None,
    cache_dir: Path | str | None = None,
) -> list[dict[str, object]]:
    openai_api_keys = itertools.cycle(openai_api_keys)
    openai_api_keys = next(openai_api_keys)

    openai_api_key = openai_api_keys[0]
    if cache_dir is not None:
        cache_dir = Path(cache_dir).expanduser().absolute()
        cache_dir.mkdir(parents=True, exist_ok=True)

    system_prompt = input['system_prompt']
    user_prompt = input['user_prompt']
    sha256 = hashlib.sha256((system_prompt + user_prompt).encode('utf-8')).hexdigest()

    if cache_dir is not None:
        output_file = cache_dir / f'{sha256}.json'
        if output_file.exists():
            with output_file.open(mode='rt', encoding='utf-8') as f:
                try:
                    return id, json.load(f)
                except json.JSONDecodeError:
                    output_file = None
    else:
        output_file = None

    if type == 'image-recognition':
        image_url = input['transformed_input']['image_url']
        messages = [
            {'role': 'system', 'content': system_prompt},
            {
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': user_prompt},
                    {
                        'type': 'image_url',
                        'image_url': {'url': f'data:image/jpeg;base64,{image_url}'},
                    },
                ],
            },
        ]

    else:

        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt},
        ]

    result = input.copy()

    result.update(
        request_openai_noexcept(
            messages=messages,
            openai_api_keys=openai_api_key,
            openai_model=openai_model,
            base_url=base_url,
        ),
    )

    result['sha256'] = sha256

    if output_file is not None:
        with output_file.open(mode='wt', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
    return id, result


def batch_request_openai(
    type: str,
    inputs: list[dict[str, Any]],
    openai_api_keys: list[str],
    openai_models: list[str],
    base_url: str | None = None,
    num_workers: int = 8,
    cache_dir: Path | str | None = None,
) -> list[dict[str, object]]:
    openai_api_keys = sorted(set(openai_api_keys))
    openai_models = sorted(set(openai_models))
    if cache_dir is not None:
        cache_dir = Path(cache_dir).expanduser().absolute()
        cache_dir.mkdir(parents=True, exist_ok=True)

    pending = deque(enumerate(inputs))
    not_ready = []
    results = [None for _ in range(len(pending))]
    openai_api_keys_cycle = itertools.cycle(
        [openai_api_keys[i:] + openai_api_keys[:i] for i in range(len(openai_api_keys))],
    )
    with tqdm.tqdm(total=len(pending)) as pbar:
        while len(not_ready) > 0 or len(pending) > 0:
            while len(not_ready) < num_workers and len(pending) > 0:
                idx, input = pending.popleft()
                current_key = next(openai_api_keys_cycle)
                not_ready.append(
                    request_openai.remote(
                        idx,
                        type,
                        input,
                        openai_api_keys=current_key,
                        openai_model=random.choice(openai_models),
                        base_url=base_url,
                        cache_dir=cache_dir,
                    ),
                )

            ready, not_ready = ray.wait(not_ready, timeout=1)
            for idx, result in ray.get(ready):
                results[idx] = result
            pbar.update(len(ready))

    return results


def get_openai_api_keys(
    openai_api_keys: list[str],
    openai_api_key_file: Path | str | None,
) -> list[str]:
    openai_api_keys = list(openai_api_keys or [])

    if openai_api_key_file is not None:
        openai_api_key_file = Path(openai_api_key_file).expanduser().absolute()
        with openai_api_key_file.open(mode='rt', encoding='utf-8') as f:
            for line in f:
                line = re.sub(r'#.*', '', line).strip()
                parts = tuple(line.split(','))
                if not line:
                    continue
                if not line.startswith('sk-'):
                    raise ValueError(f'Invalid OpenAI API key: {line}')
                openai_api_keys.append(parts)

    openai_api_keys = list(dict.fromkeys(openai_api_keys))

    if len(openai_api_keys) == 0:
        openai_api_key = os.getenv('OPENAI_API_KEY')
        if openai_api_key is not None:
            openai_api_keys.append(openai_api_key)
        else:
            raise ValueError('No OpenAI API key provided.')

    for i, [openai_api_key] in enumerate(openai_api_keys, start=1):
        if not openai_api_key.startswith('sk-'):
            raise ValueError(f'Invalid OpenAI API key: {openai_api_key}')

    return openai_api_keys


def get_annotator_response_b_prompt(data, type):

    if type == 'safety':
        question = str(data['question'])
        response_a = str(data['response_a'])
        response_b = str(data['response_b'])
        user_prompt = SAFETY_SCORE_USER_PROMPT.format(
            prompt=question,
            responseA=response_a,
            responseB=response_b,
        )
        return SAFETY_SCORE_SYSTEM_PROMPT, user_prompt
    elif type == 'utility':
        question = str(data['question'])
        response_a = str(data['response_a'])
        response_b = str(data['response_b'])
        user_prompt = UTILITY_SCORE_USER_PROMPT.format(
            prompt=question,
            responseA=response_a,
            responseB=response_b,
        )
        return UTILITY_SCORE_SYSTEM_PROMPT, user_prompt
    elif type == 'empathetic':
        question = str(data['question'])
        response_a = str(data['response_a'])
        response_b = str(data['response_b'])
        user_prompt = HELPFUL_SCORE_USER_PROMPT.format(
            prompt=question,
            responseA=response_a,
            responseB=response_b,
        )
        return HELPFUL_SCORE_SYSTEM_PROMPT, user_prompt
    elif type == 'reasoning':
        question = str(data['question'])
        response_a = str(data['response_a'])
        response_b = str(data['response_b'])
        ground_truth = str(data['target'])

        user_prompt = REASONING_SCORE_USER_PROMPT.format(
            question=question,
            gt=ground_truth,
            response_a=response_a,
            response_b=response_b,
        )
        return REASONING_SCORE_SYSTEM_PROMPT, user_prompt
    elif type == 'image-recognition':
        question = str(data['question'])
        response_a = str(data['response_a'])
        response_b = str(data['response_b'])

        user_prompt = IMAGE_RECOGNITION_USER_PROMPT.format(
            question=question,
            response_a=response_a,
            response_b=response_b,
        )
        return IMAGE_RECOGNITION_SYSTEM_PROMPT, user_prompt

    else:
        raise RuntimeError('not implemented type')


def transform_data(data, type):
    order = fixed_random(data.get('prompt', data.get('question')))
    new_data = {}
    if type == 'reasoning':
        new_data['target'] = data['target']
        new_data['question'] = data.get('prompt', data.get('question'))
        if order == 0:
            new_data['response_a'] = data.get('original_response_a', data.get('response_a'))
            new_data['response_b'] = data.get('response_b_response_a', data.get('response_b'))
        else:
            new_data['response_a'] = data.get('response_b_response_a', data.get('response_b'))
            new_data['response_b'] = data.get('original_response_a', data.get('response_a'))

        new_data['order'] = order
        return new_data, order

    if type == 'image-recognition':
        new_data['question'] = data.get('prompt', data.get('question'))
        if order == 0:
            new_data['response_a'] = data.get('original_response_a', data.get('response_a'))
            new_data['response_b'] = data.get('response_b_response_a', data.get('response_b'))
        else:
            new_data['response_a'] = data.get('response_b_response_a', data.get('response_b'))
            new_data['response_b'] = data.get('original_response_a', data.get('response_a'))

        new_data['order'] = order
        new_data['image_url'] = data.get('image_url')
        return new_data, order

    if ('category' not in data) or (not data['category']):
        violation = []
    else:
        violation = data['category']

    new_data['violation'] = violation
    new_data['question'] = data.get('prompt', data.get('question'))
    if order == 0:
        new_data['response_a'] = data.get('original_response_a', data.get('response_a'))
        new_data['response_b'] = data.get('response_b_response_a', data.get('response_b'))
    else:
        new_data['response_a'] = data.get('response_b_response_a', data.get('response_b'))
        new_data['response_b'] = data.get('original_response_a', data.get('response_a'))
    new_data['order'] = order
    return new_data, order


def prepare_inputs(
    input_file: Path | str,
    shuffle: bool = False,
    type: str = 'image-recognition',
    platform: str = 'openai',
) -> list[Any]:
    input_file = Path(input_file).expanduser().absolute()

    if input_file.suffix.lower() == '.json':
        with input_file.open(mode='rt', encoding='utf-8') as f:
            raw_inputs = json.load(f)
    elif input_file.suffix.lower() == '.jsonl':
        with input_file.open(mode='rt', encoding='utf-8') as f:
            raw_inputs = [json.loads(line) for line in f]

    inputs = []
    i = 0
    for raw_input in raw_inputs:
        data, order = transform_data(raw_input, type)
        system_prompt, user_prompt = get_annotator_response_b_prompt(data, type)
        inputs.append(
            {
                'system_prompt': system_prompt,
                'user_prompt': user_prompt,
                'transformed_input': data,
                'input_file': str(input_file),
                'order': order,
            },
        )
        i += 1

    if shuffle:
        random.shuffle(inputs)
    return inputs
