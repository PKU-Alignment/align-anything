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
import json

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
            if test_item['index'] == output_item['question_id'] and output_item['question_id'] not in question_id:
                question_id.add(output_item['question_id'])
                num_sum += 1
                if judger(test_item['answer'], output_item['response'][0]):
                    num_match += 1

    return num_match, num_sum

def judger(correct_answer, response):
    if correct_answer not in response:
        return False
    for first_response in response:
        if first_response in "ABCD":
            return first_response == correct_answer

def main():
    cache_path = ".cache"
    raw_outputs = {}

    task_dirs = [(task, os.path.join(cache_path, task)) for task in os.listdir(cache_path) if os.path.isdir(os.path.join(cache_path, task))]
    for task, task_dir in task_dirs:
        task_files = os.listdir(task_dir)
        InferenceOutputs = []
        for file in task_files:
            if file.endswith(".pkl"):
                file_path = os.path.join(task_dir, file)
                with open(file_path, 'rb') as f:
                    InferenceOutputs.extend(pickle.load(f))
        raw_outputs[task] = InferenceOutputs

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    _, unparsed_args = parser.parse_known_args()
    keys = [k[2:] for k in unparsed_args[0::2]]
    values = list(unparsed_args[1::2])
    unparsed_args = dict(zip(keys, values))
    
    dict_configs, _ = read_eval_cfgs('mmbench', 'deepspeed')
    
    try:
        assert dict_configs, "Config file does not exist or is incomplete."
    except AssertionError as e:
        logger.log('error', "Config file is not exist or incomplete.")
        exit()

    for k, v in unparsed_args.items():
        if v == '' or v is None:
            continue
        dict_configs = update_dict(dict_configs, custom_cfgs_to_dict(k, v))
    
    dict_configs = dict_to_namedtuple(dict_configs)
    data_cfgs = dict_configs.default.data_cfgs

    logger = EvalLogger('Align-Anything-Evaluation', dict_configs.default.eval_cfgs.output_dir)
    output_dicts = []
    tot_num_match, tot_num_sum = 0, 0
    for task, _ in raw_outputs.items():
        test_data = load_dataset(data_cfgs.task_dir, task)[data_cfgs.split]

        num_match, num_sum = evaluator(test_data, raw_outputs[task])
        tot_num_match += num_match
        tot_num_sum += num_sum

        output_dict = {
            'model_id': [dict_configs.default.model_cfgs.model_id],
            'num_match': [num_match],
            'num_sum': [num_sum],
            'accuracy': [num_match / num_sum]
        }
        logger.print_table(title=f'MMBench/{task} Benchmark ', data=output_dict)
        output_dicts.append(output_dict)

    output_dict = {
        'model_id': [dict_configs.default.model_cfgs.model_id],
        'tot_num_match': [tot_num_match],
        'tot_num_sum': [tot_num_sum],
        'tot_accuracy': [tot_num_match / tot_num_sum]
    }
    logger.print_table(title=f'MMBench Benchmark ', data=output_dict)
    output_dicts.append(output_dict)

    output_file = 'output_results.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_dicts, f, ensure_ascii=False, indent=4)

if __name__=="__main__":
    main()