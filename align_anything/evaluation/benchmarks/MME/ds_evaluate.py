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


def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def evaluator(test_dataset, output_data):
    num_q_match = 0
    num_q_sum = 0
    num_p_match = 0
    question_id = set()
    the_other_result = {}
    for test_item in test_dataset['test']:
        for output_item in output_data:
            if test_item['question_id'] == output_item['question_id'] and test_item['question'] in output_item['prompt_text']:
                
                num_q_sum += 1
                if judger(test_item['answer'].lower(), output_item['response'][0].lower()):
                    num_q_match += 1
                    if output_item['question_id'] not in question_id:
                        question_id.add(output_item['question_id'])
                        the_other_result[output_item['question_id']] = True
                    else:
                        num_p_match += 1
                else:
                    if output_item['question_id'] not in question_id:
                        question_id.add(output_item['question_id'])
                        the_other_result[output_item['question_id']] = False

    return num_q_match, num_q_sum, num_p_match, len(question_id)
                

def judger(target, output):
    if target not in output:
        return False
    if "yes" in output and "no" not in output:
        return target == "yes"
    if "no" in output and "yes" not in output:
        return target == "no"
    last_yes = output.rfind('yes')
    last_no = output.rfind('no')
    if last_yes > last_no:
        return target == "yes"
    elif last_no > last_yes:
        return target == "no"

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
    
    dict_configs, _ = read_eval_cfgs('mme', 'deepspeed')
    for k, v in unparsed_args.items():
        if v == '' or v is None:
            continue
        dict_configs = update_dict(dict_configs, custom_cfgs_to_dict(k, v))
    
    dict_configs = dict_to_namedtuple(dict_configs)
    data_cfgs = dict_configs.default.data_cfgs
    test_data = load_dataset(data_cfgs.task_dir, 'default')

    logger = EvalLogger('Align-Anything-Evaluation', dict_configs.default.eval_cfgs.output_dir)

    num_q_match, num_q_sum, num_pic_match, num_pic_sum = evaluator(test_data, InferenceOutputs)
    
    output_dict = {
        'model_id': [dict_configs.default.model_cfgs.model_id],
        'num_match_per_question': [num_q_match],
        'num_sum_per_question': [num_q_sum],
        'accuracy': [num_q_match / num_q_sum],
        'num_match_per_picture': [num_pic_match],
        'num_sum_per_picture': [num_pic_sum],
        'accuracy+': [num_pic_match / num_pic_sum]
    }
    logger.print_table(title='MME Benchmark', data=output_dict)


if __name__=="__main__":
    main()