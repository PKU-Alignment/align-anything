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
from align_anything.evaluation.inference.vllm_inference import *
from align_anything.evaluation.benchmarks.MME.ds_infer import MMEDataLoader
from align_anything.utils.tools import read_eval_cfgs, dict_to_namedtuple, update_dict, custom_cfgs_to_dict
from align_anything.evaluation.eval_logger import EvalLogger
from tqdm import tqdm

def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data
  
def evaluator(test_dataset, output_data, file_path):
    num_match = 0
    num_match_plus = 0
    num_sum = 0
    question_id = set()
    question_uuid = set()
    for test_item in tqdm(test_dataset, desc="Evaluating"):
        for output_item in output_data:
            if test_item['question_id'] + test_item['question'] == output_item['question_id'] and output_item['question_id'] not in question_uuid:
                question_uuid.add(output_item['question_id'])
                num_sum += 1
                true_or_false = judger(test_item['answer'].lower(), output_item['response'][0].lower())
                if true_or_false:
                    if test_item['question_id'] in question_id:
                        num_match_plus += 1
                    question_id.add(test_item['question_id'])
                    num_match += 1
                save_detail(test_item['question'], output_item['prompt_text'], test_item['answer'].lower(), output_item['response'][0].lower(), true_or_false, file_path)

    return (num_match / num_sum) * 100, (num_match_plus / (num_sum / 2)) * 100, num_sum

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
    assert os.path.exists(cache_path), ".cache folder not found. ds_infer failed?"

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

    raw_outputs = {}
    uuid_path = os.path.join(cache_path, dict_configs.default.eval_cfgs.uuid)
    assert os.path.exists(uuid_path), "uuid_path not found. ds_infer failed?"
    task_dirs = [(task, os.path.join(uuid_path, task)) for task in os.listdir(uuid_path) if os.path.isdir(os.path.join(uuid_path, task))]
    for task, task_dir in task_dirs:
        task_files = os.listdir(task_dir)
        InferenceOutputs = []
        for file in task_files:
            if file.endswith(".pkl"):
                file_path = os.path.join(task_dir, file)
                with open(file_path, 'rb') as f:
                    InferenceOutputs.extend(pickle.load(f))
        raw_outputs[task] = InferenceOutputs

    dataloader = MMEDataLoader(dict_configs)
    dataset = dataloader.get_category_datasets()
    test_data = dataloader.load_dataset(dataset)
    eval_configs = dict_configs.default.eval_cfgs

    logger = EvalLogger('Align-Anything-Evaluation', dict_configs.default.eval_cfgs.output_dir)
    logger.log_dir = eval_configs.output_dir

    os.makedirs(logger.log_dir, exist_ok=True)
    uuid_path = f"{logger.log_dir}/{eval_configs.uuid}"
    os.makedirs(uuid_path, exist_ok=True)

    tot_score, tot_num_sum = 0.0, 0
    for task, test_data in dataset.items():
        file_path = f"{uuid_path}/{task}.json"
        acc, acc_plus, num_sum = evaluator(test_data, raw_outputs[task], file_path)
        score = acc + acc_plus
        tot_score += score
        tot_num_sum += num_sum

        output_dict = {
            'model_id': [dict_configs.default.model_cfgs.model_id],
            'num_sum': [num_sum],
            'acc': [acc],
            'acc_plus': [acc_plus],
            'score': [score]
        }
        logger.print_table(title=f'MME/{task} Benchmark', data=output_dict)
        logger.log('info', '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        logger.log('info', f"task: {task}")
        logger.log('info', f"model_id: {output_dict['model_id'][0]},")
        logger.log('info', f"num_sum: {output_dict['num_sum'][0]},")
        logger.log('info', f"acc: {output_dict['acc'][0]},")
        logger.log('info', f"acc_plus: {output_dict['acc_plus'][0]},")
        logger.log('info', f"score: {output_dict['score'][0]},")
        logger.log('info', '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

    output_dict = {
        'model_id': [dict_configs.default.model_cfgs.model_id],
        'tot_num_sum': [tot_num_sum],
        'tot_score': [tot_score]
    }
    logger.print_table(title=f'MME Benchmark', data=output_dict)
    logger.log('info', '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    logger.log('info', f"model_id: {output_dict['model_id'][0]},")
    logger.log('info', f"tot_num_sum: {output_dict['tot_num_sum'][0]},")
    logger.log('info', f"tot_score: {output_dict['tot_score'][0]},")
    logger.log('info', '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
 
if __name__=="__main__":
    main()