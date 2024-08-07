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
import re

def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def evaluator(test_dataset, output_data):
    num_match = 0
    num_sum = 0
    question_id = set()
    for test_item in test_dataset:
        for output_item in output_data:
            if test_item['pid'] == output_item['question_id'] and output_item['question_id'] not in question_id:
                question_id.add(output_item['question_id'])
                num_sum += 1
                if judger(test_item['query'], test_item['answer'], output_item['response'][0]):
                    num_match += 1

    return num_match, num_sum
                
def extract_number(response):
    match = re.search(r'(?<![a-zA-Z])[A-Z](?![a-zA-Z])', response)
    if match:
        return ord(match.group().lower()) - ord('a')
    return None

def check_answer(question, response, answer):
    choice = extract_number(response)
    if "Choices:" in question and choice is not None:
        choices = re.findall(r'\(([A-Z])\)\s([^()]+)', question)
        choices_list = [choice[1].strip() for choice in choices]
        len_list = len(choices_list)
        if choice < len_list and answer == choices_list[choice]:
            return True
    return False

def judger(question, correct_answer, response):
    if correct_answer in response:
        return True
    if check_answer(question, response, correct_answer):
        return True
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
    
    dict_configs, _ = read_eval_cfgs('mathvista', 'deepspeed')
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
    logger.print_table(title='MathVista Benchmark', data=output_dict)

if __name__=="__main__":
    main()