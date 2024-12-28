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

import hashlib
import json
import logging
import os
import time
from typing import Any, Callable

import pkg_resources
from packaging import version


openai_version = pkg_resources.get_distribution('openai').version
new_openai_flag = version.parse(openai_version) >= version.parse('1.0.0')
if not new_openai_flag:
    pass

API_KEY = ''

import ray
import urllib3
from tqdm import tqdm
from urllib3.util.retry import Retry


@ray.remote(num_cpus=1)
def bean_gpt_api(
    system_content: str,
    user_content: str,
    post_process: Callable = lambda x: x,
) -> Any:
    """Baichuan GPT API"""
    messages = [
        {'role': 'system', 'content': system_content},
        {'role': 'user', 'content': user_content},
    ]

    openai_api = 'https://api.openai.com'

    params_gpt = {
        'model': 'gpt-4o',
        'messages': messages,
        'temperature': 0.5,
    }
    url = openai_api + '/v1/chat/completions'

    headers = {
        'Content-Type': 'application/json',
        'Authorization': API_KEY,
        'Connection': 'close',
    }

    retry_strategy = Retry(
        total=5,  # Maximum retry count
        backoff_factor=0.1,  # Wait factor between retries
        status_forcelist=[429, 500, 502, 503, 504],  # HTTP status codes to force a retry on
        allowed_methods=['POST'],  # Retry only for POST request
        raise_on_redirect=False,  # Don't raise exception
        raise_on_status=False,  # Don't raise exception
    )
    http = urllib3.PoolManager(
        retries=retry_strategy,
    )
    encoded_data = json.dumps(params_gpt).encode('utf-8')
    max_try = 1000
    while max_try > 0:
        try:
            response = http.request('POST', url, body=encoded_data, headers=headers)
            if response.status == 200:
                response = json.loads(response.data.decode('utf-8'))['choices'][0]['message'][
                    'content'
                ]
                logging.info(response)
                break
            else:
                err_msg = f'Access openai error, status code: {response.status} response: {response.data.decode("utf-8")}'
                logging.error(err_msg)
                time.sleep(3)
                max_try -= 1
                continue
        except:
            err_msg = f'Access openai error, status code: {response.status} response: {response.data.decode("utf-8")}'
            logging.error(err_msg)
            time.sleep(3)
            max_try -= 1
            continue
    else:
        response = 'Bean Proxy API Failed...'

    return post_process(response)


def generate_hash_uid(to_hash: dict | tuple | list | str):
    """Generates a unique hash for a given model and arguments."""
    # Convert the dictionary to a JSON string
    json_string = json.dumps(to_hash, sort_keys=True)

    # Generate a hash of the JSON string
    hash_object = hashlib.sha256(json_string.encode())
    hash_uid = hash_object.hexdigest()

    return hash_uid


def api(
    system_contents: list[str],
    user_contents: list[str],
    num_workers: int = 50,
    post_process: Callable = lambda x: x,
    hash_checker: Callable = lambda x: True,
    cache_dir: str = './cache',
):
    """API"""
    if len(system_contents) != len(user_contents):
        raise ValueError('Length of system_contents and user_contents should be equal.')
    server = bean_gpt_api

    api_interaction_count = 0
    ray.init()

    contents = list(enumerate(zip(system_contents, user_contents)))
    bar = tqdm(total=len(system_contents))
    results = [None] * len(system_contents)
    uids = [generate_hash_uid(content) for content in contents]
    not_finished = []
    while True:
        if len(not_finished) == 0 and len(contents) == 0:
            break
        while len(not_finished) < num_workers and len(contents) > 0:
            index, content = contents.pop()
            uid = uids[index]
            cache_path = os.path.join(cache_dir, f'{uid}.json')
            if os.path.exists(cache_path):
                with open(cache_path, encoding='utf-8') as f:
                    try:
                        result = json.load(f)
                    except:
                        os.remove(cache_path)
                        print(cache_path)
                        continue
                results[index] = result
                bar.update(1)
                continue

            future = server.remote(content[0], content[1], post_process)
            not_finished.append([index, future])
            api_interaction_count += 1

        if len(not_finished) == 0:
            continue
        indices, futures = zip(*not_finished)

        finished, not_finished_futures = ray.wait(list(futures), timeout=1.0)
        finished_indices = [indices[futures.index(task)] for task in finished]

        for i, task in enumerate(finished):
            results[finished_indices[i]] = ray.get(task)
            uid = uids[finished_indices[i]]
            cache_path = os.path.join(cache_dir, f'{uid}.json')
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(results[finished_indices[i]], f, ensure_ascii=False, indent=4)
        not_finished = [(index, future) for index, future in not_finished if future not in finished]

        bar.update(len(finished))
    bar.close()

    assert all(result is not None for result in results)

    ray.shutdown()
    print(f'API interaction count: {api_interaction_count}')

    return results
