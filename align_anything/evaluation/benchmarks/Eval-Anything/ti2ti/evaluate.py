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
import os
import re
import time
import io
import ray
import json
import base64
import hashlib
import logging
import urllib3
import argparse
import pkg_resources
from tqdm import tqdm
from PIL import Image
from typing import Any, Callable
from packaging import version
from urllib3.util.retry import Retry
from align_anything.evaluation.eval_logger import EvalLogger
from align_anything.utils.tools import read_eval_cfgs, dict_to_namedtuple, update_dict, custom_cfgs_to_dict
from prompt import system_prompt, user_prompt_1, user_prompt_2 
openai_version = pkg_resources.get_distribution('openai').version
new_openai_flag = version.parse(openai_version) >= version.parse("1.0.0")
if not new_openai_flag:
    import openai

@ray.remote(num_cpus=1)
def gpt_judger(
    api_key: str,
    base_url: str,
    system_content: str,
    user_content_1: str,
    user_content_2: str,
    image_url_1: str | None = None,
    image_url_2: str | None = None,
    post_process: Callable = lambda x: x,
) -> Any:
    messages = [
        {'role': 'system', 'content': system_content},
        {
            'role': 'user',
            "content": [
                {"type": "text", "text": user_content_1},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_url_1}",
                    },
                },
                {"type": "text", "text": user_content_2},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_url_2}",
                    },
                },
            ],
        },
    ]

    params_gpt = {
        'model': 'gpt-4o',
        'messages': messages,
        'temperature': 0.05,
    }
    headers = {
        'Content-Type': 'application/json',
        'Authorization': api_key,
        'Connection':'close',
        }

    retry_strategy = Retry(
        total=5,
        backoff_factor=0.1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=['POST'],
        raise_on_redirect=False,
        raise_on_status=False, 
    )
    http = urllib3.PoolManager(
        retries=retry_strategy,
    )
    encoded_data = json.dumps(params_gpt).encode('utf-8')
    max_try = 1000
    while max_try > 0:
        try:
            response = http.request('POST', base_url, body=encoded_data, headers=headers)
            if response.status == 200:
                    response = json.loads(response.data.decode('utf-8'))['choices'][0]['message']['content']
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
        print('Bean Proxy API Failed...')
        response = 'Bean Proxy API Failed...'

    return post_process(response)

def generate_hash_uid(to_hash: dict | tuple | list | str):
    json_string = json.dumps(to_hash, sort_keys=True)

    hash_object = hashlib.sha256(json_string.encode())
    hash_uid = hash_object.hexdigest()

    return hash_uid

def get_result(
    api_key: str,
    base_url: str,
    system_contents: list[str],
    user_contents_1: list[str],
    user_contents_2: list[str],
    image_urls_1: list[str] | None = None,
    image_urls_2: list[str] | None = None,
    num_workers: int = 50,
    post_process: Callable = lambda x: x,
):
    if len(system_contents) != len(user_contents_1):
        raise ValueError('Length of system_contents and user_contents should be equal.')
    server = gpt_judger

    api_interaction_count = 0
    ray.init()

    contents = list(enumerate(zip(system_contents, user_contents_1, user_contents_2, image_urls_1, image_urls_2)))
    bar = tqdm(total=len(system_contents))
    results = [None] * len(system_contents)
    not_finished = []
    while True:
        if len(not_finished) == 0 and len(contents) == 0:
            break

        while len(not_finished) < num_workers and len(contents) > 0:
            index, content = contents.pop()
            future = server.remote(api_key, base_url, content[0], content[1], content[2], content[3], content[4], post_process)
            not_finished.append([index, future])
            api_interaction_count += 1

        if len(not_finished) == 0:
            continue

        indices, futures = zip(*not_finished)
        finished, not_finished_futures = ray.wait(list(futures), timeout=1.0)
        finished_indices = [indices[futures.index(task)] for task in finished]

        for i, task in enumerate(finished):
            results[finished_indices[i]] = ray.get(task)

        not_finished = [(index, future) for index, future in not_finished if future not in finished]
        bar.update(len(finished))

    bar.close()
    assert all(result is not None for result in results)
    ray.shutdown()

    return results

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        img = Image.open(image_file)
        img.thumbnail((512, 512))
        
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        img_bytes = buffered.getvalue()
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')
    
    return img_base64

def get_sorted_images_from_path(image_path):
    image_files = []
    for root, dirs, files in os.walk(image_path):
        for file in files:
            if file.lower().endswith('.png'):
                match = re.search(r'(\d+)', file)
                if match:
                    image_files.append((int(match.group(1)), os.path.join(root, file)))
    
    image_files.sort(key=lambda x: x[0])
    
    return [encode_image(file[1]) for file in image_files], [file[1] for file in image_files]

def get_datasets(data):   
    final_dataset = []
    ori_paths_1 = []
    processed_paths_1 = []
    ori_paths_2 = []
    processed_paths_2 = []
    
    for item in tqdm(data, desc="Processing Entries"):
        ori_path_1 = item['input_image_url']
        processed_paths_1.append(encode_image(ori_path_1))
        ori_paths_1.append(ori_path_1)

        ori_path_2 = item['responses']['output_image_url']
        processed_paths_2.append(encode_image(ori_path_2))
        ori_paths_2.append(ori_path_2)
        
        final_item = {
            'question': item.get('question', 'NULL'),
            'text_response': item['responses']['text_response'],
        }
        final_dataset.append(final_item)
    
    return processed_paths_1, ori_paths_1, processed_paths_2, ori_paths_2, final_dataset

def extract_results_output(input_string):
    keys = re.findall(r'\[\[(.*?)\]\]', input_string)
    
    values = re.split(r'\[\[(?:.*?)\]\]', input_string)[1:]
    values = [value.strip() for value in values]
    
    result = dict(zip(keys, values))
    
    return result

def get_score(data):
    match = re.search(r'\d+', data)
    score = int(match.group()) if match else 0
    return score

def judger(api_key, base_url, dataset, file_path):
    processed_paths_1, ori_paths_1, processed_paths_2, ori_paths_2, final_dataset = get_datasets(dataset)
    
    def post_process(response: str):
        return response
    
    system_prompts = []
    
    user_prompts_q = [user_prompt_1.format(prompt=item['question']) for item in final_dataset]
    user_prompts_a1 = [user_prompt_2.format(text=item['text_response']) for item in final_dataset]

    system_prompts = [system_prompt] * len(processed_paths_1)
        
    results = get_result(api_key, base_url, system_prompts, user_prompts_q, user_prompts_a1, processed_paths_1, processed_paths_2, post_process=post_process)
    
    num_sum = 0
    datas = []
    prompt_following_rate, objective_rules_rate, ca_rate, information_richness_rate, safety_rate, consistency_rate = 0, 0, 0, 0, 0, 0
    for i, item in enumerate(final_dataset):
        num_sum += 1
        result = extract_results_output(results[i])
        rates = {
            'prompt_following_rate': result.get('P_Rate', 'NULL'),
            'objective_rules_rate': result.get('O_Rate', 'NULL'),
            'ca_rate': result.get('CA_Rate', 'NULL'),
            'information_richness_rate': result.get('I_Rate', 'NULL'),
            'safety_rate': result.get('S_Rate', 'NULL'),
            'consistency_rate': result.get('C_Rate', 'NULL'),
        }
        prompt_following_rate += get_score(rates['prompt_following_rate'])
        objective_rules_rate += get_score(rates['objective_rules_rate'])
        ca_rate += get_score(rates['ca_rate'])
        information_richness_rate += get_score(rates['information_richness_rate'])
        safety_rate += get_score(rates['safety_rate'])
        consistency_rate += get_score(rates['consistency_rate'])

        data = {**item, **result}
        datas.append(data)

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(datas, f, ensure_ascii=False, indent=4)

    return round(prompt_following_rate / num_sum, 2), round(objective_rules_rate / num_sum, 2), round(ca_rate / num_sum, 2), round(information_richness_rate / num_sum, 2), round(safety_rate / num_sum, 2), round(consistency_rate / num_sum, 2), num_sum

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    _, unparsed_args = parser.parse_known_args()
    keys = [k[2:] for k in unparsed_args[0::2]]
    values = list(unparsed_args[1::2])
    unparsed_args = dict(zip(keys, values))
    
    dict_configs, _ = read_eval_cfgs('eval-anything', 'deepspeed')
    for k, v in unparsed_args.items():
        if v == '' or v is None:
            continue
        dict_configs = update_dict(dict_configs, custom_cfgs_to_dict(k, v))
    
    dict_configs = dict_to_namedtuple(dict_configs)

    eval_configs = dict_configs.default.eval_cfgs

    logger = EvalLogger('Align-Anything-Evaluation', dict_configs.default.eval_cfgs.output_dir)
    logger.log_dir = eval_configs.output_dir

    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_API_BASE_URL")
    
    if not api_key:
        raise ValueError("OpenAI API key is not provided in eval_configs or environment variables.")
    if not base_url:
        raise ValueError("OpenAI API base URL is not provided in eval_configs or environment variables.")

    os.makedirs(logger.log_dir, exist_ok=True)
    uuid_path = f"{logger.log_dir}/{eval_configs.uuid}"
    os.makedirs(uuid_path, exist_ok=True)

    dataset_path = "./input.json"
    assert os.path.exists(dataset_path), f"The file '{dataset_path}' does not exist."
    with open(dataset_path, 'r') as file:
        dataset = json.load(file)

    file_path = f"{uuid_path}/default.json"
    prompt_following_score, objective_rules_score, clarity_aesthetics_score, information_richness_score, safety_score, consistency_score, num_sum = judger(api_key, base_url, dataset, file_path)

    output_dict = {
        'model_id': [dict_configs.default.model_cfgs.model_id],
        'num_sum': [num_sum],
        'prompt_following_score': [f'{prompt_following_score}/3'],
        'objective_rules': [f'{objective_rules_score}/3'],
        'clarity_aesthetics': [f'{clarity_aesthetics_score}/3'],
        'information_richness': [f'{information_richness_score}/3'],
        'safety': [f'{safety_score}/3'],
        'consistency': [f'{consistency_score}/3']
    }
    logger.print_table(title=f'Eval-Anything Benchmark', data=output_dict)
    logger.log('info', '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    logger.log('info', f"model_id: {output_dict['model_id'][0]},")
    logger.log('info', f"num_sum: {output_dict['num_sum'][0]},")
    logger.log('info', f"prompt_following_score: {output_dict['prompt_following_score'][0]},")
    logger.log('info', f"objective_rules: {output_dict['objective_rules'][0]},")
    logger.log('info', f"clarity_aesthetics: {output_dict['clarity_aesthetics'][0]},")
    logger.log('info', f"information_richness: {output_dict['information_richness'][0]},")
    logger.log('info', f"safety: {output_dict['safety'][0]},")
    logger.log('info', f"consistency: {output_dict['consistency'][0]},")
    logger.log('info', '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

if __name__ == '__main__':
    main()
