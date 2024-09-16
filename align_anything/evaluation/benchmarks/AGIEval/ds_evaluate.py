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
import requests
import re

gpt_system_prompt = """
You are an expert in evaluating responses to questions. You will be given a question, the correct answer, and a response provided by someone else. Your task is to determine whether the response is correct based on the correct answer. 
Please carefully analyze the information provided and answer with "True" if the response is correct or "False" if the response is incorrect.

Question: {INSERT_PROMPT_HERE}
Correct Answer: {INSERT_CORRECT_ANSWER_HERE}
Response: {INSERT_TEXT_OF_RESPONSE_HERE}
Is the response correct? Please answer with "True" or "False".
"""
class AGIEvalDataLoader(BaseDataLoader):
    def get_task_names(self):
        if isinstance(self.data_cfgs.task, list):
            return self.data_cfgs.task
        else:
            task_names = [
            self.data_cfgs.task
            ]
            return task_names

    def get_answer(self, data):
        if (isinstance(data['answer'], list)):
            data['answer'] = data['answer'][0]
        return data['answer']

    def build_example_prompt(self, data, with_answer=True, cot=False):
        choices = ''
        if data["choices"]:
            choices = '\n'.join([f'{data["choices"][label]}' for label in range(len(data["choices"]))])
        answer = f'Answer: {self.get_answer(data)}' if with_answer else 'Answer: '
        return f"{data['passage']}\n{data['question']}Please choose the correct answer from the following options:\n{choices}\n{answer}"

    def build_prompt(self, data):
        prompt = ""
        template = get_template_class(self.chat_template)
        question = [template.system_prompt + template.user_prompt.format(input=prompt + self.build_example_prompt(item, False)) + template.assistant_prompt.format(output="") for item in data]

        return question
    
def evaluator(raw_output: List[InferenceOutput], dataloader: AGIEvalDataLoader, task: str, api_key, base_url, file_path):
    dataset = load_dataset(dataloader.task_dir, task)[dataloader.split]
    correct_answers = []
    responses = []
    cnt_sum = 0
    cnt_match = 0
    for instance in dataset:
        correct_answers.append(
            {
                'prompt': instance['question'],
                'choices': instance['choices'],
                'answer': dataloader.get_answer(instance)
            }
        )
    for item in raw_output:
        responses.append(
            {
                'prompt': (item.prompt),
                'answer': item.response[0]
            }
        )
    for correct_answer in tqdm(correct_answers, desc="Evaluating"):
        cnt_sum += 1
        for response in responses:
            if correct_answer['prompt'] in response['prompt']:
                if correct_answer['choices']:
                    if isinstance(correct_answer['answer'], list):
                        correct_answer['answer'] = correct_answer['answer'][0]
                    true_or_false = judge_answer(correct_answer['answer'], response['answer'])
                    choices = '\n' + '\n'.join([f'({chr(label + 65)}) {correct_answer["choices"][label]}' for label in range(len(correct_answer["choices"]))])
                else:
                    true_or_false = gpt_judge_answer(correct_answer['prompt'], correct_answer['answer'], response['answer'], api_key, base_url)
                    choices = ''
                if true_or_false:
                    cnt_match += 1
                save_detail(correct_answer['prompt'], choices, correct_answer['answer'], response['answer'], true_or_false, file_path)
                break

    return cnt_match, cnt_sum

def gpt_judge_answer(question, correct_answer, response, api_key, base_url):
    def get_response(prompt):
        data = {
            "model": "gpt-4-turbo",
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }
        response = requests.post(
            base_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json=data
        )
        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['message']['content']
        else:
            raise Exception(f"Request failed: {response.status_code}, {response.text}")

    prompt = gpt_system_prompt.format(
        INSERT_PROMPT_HERE=question,
        INSERT_CORRECT_ANSWER_HERE=correct_answer,
        INSERT_TEXT_OF_RESPONSE_HERE=response
    )

    true_or_false = get_response(prompt)
    if true_or_false.strip().lower() == "true":
        return True
    else:
        return False
    
def judge_answer(correct_answer, response):
    match = re.search(r'(?<![a-zA-Z])[A-Z](?![a-zA-Z])', response)
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

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    _, unparsed_args = parser.parse_known_args()
    keys = [k[2:] for k in unparsed_args[0::2]]
    values = list(unparsed_args[1::2])
    unparsed_args = dict(zip(keys, values))
    logger = EvalLogger('Evaluation')

    dict_configs, _ = read_eval_cfgs('agieval', 'deepspeed')

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
    dataloader = AGIEvalDataLoader(dict_configs)

    api_key = eval_configs.openai_api_key or os.getenv("OPENAI_API_KEY")
    base_url = eval_configs.openai_api_base_url or os.getenv("OPENAI_API_BASE_URL")
    
    if not api_key:
        raise ValueError("OpenAI API key is not provided in eval_configs or environment variables.")
    if not base_url:
        raise ValueError("OpenAI API base URL is not provided in eval_configs or environment variables.")

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
        cnt_match, cnt_sum = evaluator(raw_outputs, dataloader, task, api_key, base_url, file_path)
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
        logger.print_table(title=f'AGIEval/{task} Benchmark', data=eval_results)
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
    logger.print_table(title=f'AGIEval Benchmark', data=eval_results)
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
