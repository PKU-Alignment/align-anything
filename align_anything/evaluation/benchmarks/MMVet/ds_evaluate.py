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

import pickle
import os
from datasets import load_dataset
import argparse
from align_anything.utils.tools import read_eval_cfgs, dict_to_namedtuple, update_dict, custom_cfgs_to_dict
from align_anything.evaluation.eval_logger import EvalLogger
import requests
import time

API_KEY = ""
BASE_URL = ""

def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def gpt4_judger(question, answer1, answer2):
    def get_response(prompt):
        data = {
            "model": "gpt-4-turbo",
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }
        response = requests.post(
            BASE_URL,
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json"
            },
            json=data
        )
        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['message']['content']
        else:
            raise Exception(f"Request failed: {response.status_code}, {response.text}")

    prompt = f"For the following question, please determine whether Answer_1 and Answer_2 have the same meaning\nQuestion: {question}\nnAnswer1: {answer1}\nAnswer_2: {answer2}\nPlease answer yes or no."
    result = get_response(prompt)
    
    if 'yes' in result.lower():
        return True
    elif 'no' in result.lower():
        return False

def evaluator(test_dataset, output_data):
    num_match = 0
    num_sum = 0
    question_id = set()
    for test_item in test_dataset:
        for output_item in output_data:
            if test_item['question_id'] == output_item['question_id'] and output_item['question_id'] not in question_id:
                question_id.add(output_item['question_id'])
                time.sleep(0.01)
                num_sum += 1
                if judger(test_item['question'], test_item['answer'].lower(), output_item['response'][0].lower()):
                    num_match += 1

    return num_match, num_sum

def judger(question, correct_answer, response):
    if '<and>' in correct_answer:
        and_parts = correct_answer.split('<and>')
        if all(part in response for part in and_parts):
            return True
    elif '<or>' in correct_answer:
        or_parts = correct_answer.split('<or>')
        if any(part in response for part in or_parts):
            return True
    else:
        if correct_answer in response:
            return True
        return gpt4_judger(question, correct_answer, response)

    return False
    
def main():
    cache_path = ".cache"
    files = os.listdir(cache_path)
    InferenceOutputs = []
    
    for file in files:
        if file.endswith(".pkl"):
            InferenceOutputs.extend(load_pickle(os.path.join(cache_path, file)))
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    _, unparsed_args = parser.parse_known_args()
    keys = [k[2:] for k in unparsed_args[0::2]]
    values = list(unparsed_args[1::2])
    unparsed_args = dict(zip(keys, values))
    
    dict_configs, _ = read_eval_cfgs('mmvet', 'deepspeed')
    for k, v in unparsed_args.items():
        if v == '' or v is None:
            continue
        dict_configs = update_dict(dict_configs, custom_cfgs_to_dict(k, v))
    
    dict_configs = dict_to_namedtuple(dict_configs)
    data_cfgs = dict_configs.default.data_cfgs
    test_data = load_dataset(data_cfgs.task_dir, 'default')[data_cfgs.split]

    logger = EvalLogger('Align-Anything-Evaluation', dict_configs.default.eval_cfgs.output_dir)

    num_match, num_sum = evaluator(test_data, InferenceOutputs)
    
    output_dict = {
        'model_id': [dict_configs.default.model_cfgs.model_id],
        'num_match_question': [num_match],
        'num_sum_question': [num_sum],
        'accuracy': [num_match / num_sum]
    }
    logger.print_table(title='MMVet Benchmark', data=output_dict)

if __name__=="__main__":
    main()