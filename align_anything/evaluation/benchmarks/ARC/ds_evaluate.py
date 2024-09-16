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

import os
import pickle
import argparse
import json
from align_anything.evaluation.inference.vllm_inference import *
from align_anything.evaluation.dataloader.base_dataloader import BaseDataLoader
from typing import List, Dict, Any
from datasets import load_dataset
from align_anything.utils.tools import read_eval_cfgs, dict_to_namedtuple, update_dict, custom_cfgs_to_dict
from align_anything.utils.template_registry import get_template_class
from align_anything.evaluation.data_type import InferenceOutput
from align_anything.evaluation.eval_logger import EvalLogger
import re

class ARCDataLoader(BaseDataLoader):
    def get_task_names(self):
        if isinstance(self.data_cfgs.task, list):
            return self.data_cfgs.task
        else:
            task_names = [
            self.data_cfgs.task
            ]
            return task_names

    def get_answer(self, data):
        return data['answerKey']

    def set_fewshot_dataset(self, dataset, task): 
        return dataset['validation']
        
    def build_example_prompt(self, data, with_answer=True, cot=False):
        choices_text = []
        for label in data['choices']['label']:
            if ord('A') <= ord(label) <= ord('Z'):
                choice_id = label
                choices_text.append(f'({choice_id}): {data["choices"]["text"][ord(choice_id) - 65]}' )
            else:
                choice_id = int(label)
                choices_text.append(f'({choice_id}): {data["choices"]["text"][choice_id - 1]}' )
        choices = '\n'.join(choices_text)
        answer = f'Answer: {self.get_answer(data)}' if with_answer else 'Answer: '
        return f"{data['question']}\n{choices}\n{answer}"

    def build_prompt(self, data):
        prompt = f"The following are multiple choice questions (with answers).\n\n"
        cot_prompt = f" Let's think step by step. "
        few_shot_examples = self.few_shot_data[:self.num_shot] if self.num_shot else []
        template = get_template_class(self.chat_template)
        if len(few_shot_examples) == 0:
            question = [template.system_prompt + template.user_prompt.format(input=prompt + self.build_example_prompt(item, False)) + template.assistant_prompt.format(output="") for item in data]
        else:
            if not self.cot:
                few_shots = [
                    self.build_example_prompt(
                        {key: value[i] for key, value in few_shot_examples.items()}, True
                    )
                    for i in range(len(few_shot_examples['question']))
                ]
            else:
                few_shots = [
                    f"{example['question']}\n'Answer: '{example['answer']}" for example in few_shot_examples
                ]
            question = []
            for item in data:
                request = {}
                for key, value in item.items():
                    request[key] = value
                examples = few_shots + [self.build_example_prompt(request, False)]
                if self.cot:
                    question.append(template.system_prompt + template.user_prompt.format(input=prompt + '\n\n'.join(examples)) + template.assistant_prompt.format(output=cot_prompt))
                else:
                    question.append(template.system_prompt + template.user_prompt.format(input=prompt + '\n\n'.join(examples)) + template.assistant_prompt.format(output=""))
        
        return question
    
def evaluator(raw_output: List[InferenceOutput], dataloader: ARCDataLoader, task: str, file_path):
    dataset = load_dataset(dataloader.task_dir, task)[dataloader.split]
    correct_answers = []
    responses = []
    cnt_sum = 0
    cnt_match = 0
    cnt_fail = 0
    flag_fail = True
    for instance in dataset:
        correct_answers.append(
            {
                'prompt': instance['question'],
                'choices': instance['choices'],
                'answer': dataloader.get_answer(instance)
            }
        )
    for item in raw_output:
        dataloader.candidate_labels = get_candidate_labels(item.prompt)
        responses.append(
            {
                'prompt': (item.prompt),
                'answer_logprobs': get_chosen_answer(item.response_logprobs[-1], dataloader.candidate_labels),
                'answer': item.response[0]
            }
        )
    for correct_answer in correct_answers:
        cnt_sum += 1
        for response in responses:
            if correct_answer['prompt'] in response['prompt']:
                flag_fail = False
                chosen_answer = max(response['answer_logprobs'], key=response['answer_logprobs'].get)
                eval_case = {
                    'question': correct_answer['prompt'],
                    'choices': correct_answer['choices'],
                    'correct_answer': correct_answer['answer'],
                    'answer_logprobs': response['answer_logprobs'],
                    'chosen_answer': chosen_answer
                }
                true_or_false = judge_answer(correct_answer['answer'], chosen_answer, response['answer'])
                if true_or_false:
                    cnt_match += 1
                save_detail(correct_answer['prompt'], get_choices(correct_answer), correct_answer['answer'], response['answer'], true_or_false, file_path)
                break
        if flag_fail:
            cnt_fail += 1
        else:
            flag_fail = True
        
    return cnt_match, cnt_sum

def get_choices(data):
    if data['choices']['label'][0] == 'A':
        choices = '\n' + '\n'.join([f"({chr(label+65)}) {data['choices']['text'][label]}" for label in range(len(data['choices']['text']))])
    else:
        choices = '\n' + '\n'.join([f"({label}) {data['choices']['text'][label]}" for label in range(len(data['choices']['text']))])
    return choices

def get_chosen_answer(logprobs: List[Dict[str, Any]], candidate_answers: List[str]):
    answer_logprobs = {}
    for logprob in logprobs:
        key = next(iter(logprob.values())).decoded_token
        value = next(iter(logprob.values())).logprob
        if key in candidate_answers:
            answer_logprobs[key] = value
    for label in candidate_answers:
        if label not in answer_logprobs.keys():
            answer_logprobs[label] = float('-inf')
    return answer_logprobs

def get_candidate_labels(prompt):
    number_matches = re.findall(r'\([1234]\)', prompt)
    number_index = all(option in number_matches for option in ['(1)', '(2)', '(3)', '(4)'])
    
    if number_index:
        return ['1', '2', '3', '4']
    return ['A', 'B', 'C', 'D']
    
def judge_answer(correct_answer, chosen_answer, response):
    if correct_answer == chosen_answer:
        return True
    if correct_answer in ['A', 'B', 'C', 'D']:
        match = re.search(r'(?<![a-zA-Z])[A-Z](?![a-zA-Z])', response)
        if match:
            return correct_answer == match.group()
    else:
        match = re.search(r'(?<!\d)\d(?!\d)', response)
        if match:
            return correct_answer == match.group()
    return False

def get_data(task_dir):
    if not os.path.isdir(task_dir):
        return None
    task_files = os.listdir(task_dir)
    InferenceOutputs = []
    for file in tqdm(task_files, desc='Loading data'):
        if file.endswith(".pkl"):
            file_path = os.path.join(task_dir, file)
            with open(file_path, 'rb') as f:
                InferenceOutputs.extend(pickle.load(f))
    return InferenceOutputs

def main():
    cache_path = ".cache"
    assert os.path.exists(cache_path), ".cache folder not found. ds_infer failed?"
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
    logger = EvalLogger('Evaluation')

    dict_configs, _ = read_eval_cfgs('arc', 'deepspeed')

    try:
        assert dict_configs, "Config file does not exist or is incomplete."
    except AssertionError as e:
        print("Config file is not exist or incomplete.")
        exit()

    for k, v in unparsed_args.items():
        if v == '' or v is None:
            continue
        dict_configs = update_dict(dict_configs, custom_cfgs_to_dict(k, v))

    dict_configs = dict_to_namedtuple(dict_configs)
    eval_configs = dict_configs.default.eval_cfgs
    logger.log_dir = eval_configs.output_dir
    dataloader = ARCDataLoader(dict_configs)

    os.makedirs(logger.log_dir, exist_ok=True)
    uuid_path = f"{logger.log_dir}/{eval_configs.uuid}"
    os.makedirs(uuid_path, exist_ok=True)

    tot_num_match, tot_num_sum = 0, 0
    for task in dataloader.task_names:
        task_dir = os.path.join(cache_path, task)
        raw_outputs = get_data(task_dir)
        if not raw_outputs:
            continue
        
        file_path = f"{uuid_path}/{task}.json"
        cnt_match, cnt_sum = evaluator(raw_outputs, dataloader, task, file_path)
        tot_num_match += cnt_match
        tot_num_sum += cnt_sum
        
        eval_results = {
            'model_id': [dict_configs.default.model_cfgs.model_id],
            'num_fewshot': [eval_configs.n_shot],
            'chain_of_thought': [eval_configs.cot],
            'num_match': [cnt_match],
            'num_sum': [cnt_sum],
            'accuracy': [cnt_match / cnt_sum]
        }
        logger.print_table(title=f'ARC/{task} Benchmark', data=eval_results)
        logger.log('info', '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        logger.log('info', f"task: {task}")
        logger.log('info', f"model_id: {eval_results['model_id'][0]},")
        logger.log('info', f"num_fewshot: {eval_results['num_fewshot'][0]},")
        logger.log('info', f"chain_of_thought: {eval_results['chain_of_thought'][0]},")
        logger.log('info', f"num_match: {eval_results['num_match'][0]},")
        logger.log('info', f"num_sum: {eval_results['num_sum'][0]},")
        logger.log('info', f"accuracy: {eval_results['accuracy'][0]},")
        logger.log('info', '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

    eval_results = {
        'model_id': [dict_configs.default.model_cfgs.model_id],
        'num_fewshot': [eval_configs.n_shot],
        'chain_of_thought': [eval_configs.cot],
        'tot_num_match': [tot_num_match],
        'tot_num_sum': [tot_num_sum],
        'tot_accuracy': [tot_num_match / tot_num_sum]
    }
    logger.print_table(title=f'ARC Benchmark', data=eval_results)
    logger.log('info', '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    logger.log('info', f"model_id: {eval_results['model_id'][0]},")
    logger.log('info', f"num_fewshot: {eval_results['num_fewshot'][0]},")
    logger.log('info', f"chain_of_thought: {eval_results['chain_of_thought'][0]},")
    logger.log('info', f"tot_num_match: {eval_results['tot_num_match'][0]},")
    logger.log('info', f"tot_num_sum: {eval_results['tot_num_sum'][0]},")
    logger.log('info', f"tot_accuracy: {eval_results['tot_accuracy'][0]},")
    logger.log('info', '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

if __name__=="__main__":
    main()