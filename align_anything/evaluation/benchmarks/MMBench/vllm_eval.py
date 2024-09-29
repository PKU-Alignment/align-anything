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
import json
from align_anything.evaluation.inference.vllm_inference import *
from align_anything.evaluation.dataloader.base_dataloader import BaseDataLoader
from typing import List, Dict, Any
from datasets import load_dataset, DatasetDict
from align_anything.utils.tools import read_eval_cfgs, dict_to_namedtuple, update_dict, custom_cfgs_to_dict, save_raw_outputs, load_raw_outputs
from align_anything.utils.template_registry import get_template_class
from align_anything.evaluation.data_type import InferenceInput, InferenceOutput
from align_anything.evaluation.eval_logger import EvalLogger
from tqdm import tqdm
import requests
import os
import re

gpt_system_prompt = """
You are tasked with matching a model's response to one of the provided multiple-choice options. You will be given a question, several options, and a response from the model. Your job is to determine which option (A, B, C, or D) the model's response aligns with. If the model's response does not align with any option, return the letter 'Z'.

Here is the input format:

Question: {INSERT_PROMPT_HERE}
Choices:
{INSERT_CHOICES_HERE}
The model's response: {INSERT_TEXT_OF_RESPONSE_HERE}

Please return one of the following:
- A, if the response aligns with choice A
- B, if the response aligns with choice B
- C, if the response aligns with choice C
- D, if the response aligns with choice D
- Z, if the response does not match any of the choices
"""
class MMBenchDataLoader(BaseDataLoader):
    def get_task_names(self):
        if isinstance(self.data_cfgs.task, list):
            return self.data_cfgs.task
        else:
            task_names = [
            self.data_cfgs.task
            ]
            return task_names

    def get_answer(self, data):
        return data['answer']
        
    def build_example_prompt(self, data, with_answer=True):
        choices = '(A) ' + data["A"] + '\n(B) ' + data["B"] + '\n(C) ' + data["C"] + '\n(D) ' + data["D"]
        answer = f'Answer: ({self.get_answer(data)})' if with_answer else 'Answer: '
        return f"{data['hint']}\n{data['question']}Please choose the correct answer from the following options:\n{choices}\n{answer}"

    def build_prompt(self, data):
        assert self.num_shot == 0, "MMBench does not support few-shot learning."
        prompt = f"The following are hints and multiple choice questions (with answers).\n\n"
        template = get_template_class(self.chat_template)
        question = [template.system_prompt + template.user_prompt.format(input=prompt + self.build_example_prompt(item, False)) + template.assistant_prompt.format(output="") for item in data]

        return question
    
    def preprocess(self, data):
        return self.build_prompt(data)
    
    def load_dataset(self) -> DatasetDict:
        processed_inputs = {}
        for task in self.task_names:
            dataset = load_dataset(self.task_dir, task)[self.split]
            prompts = self.preprocess(dataset)
            processed_inputs[task] = []
            for prompt, image, question_id in zip(prompts, dataset['image'], dataset['index']):
                processed_input = InferenceInput(text=prompt, image_file=image)
                processed_input.question_id = question_id
                processed_inputs[task].append(processed_input)
        return processed_inputs
    
class MMBenchGeneratorVLLM(BaseInferencer_vllm):
    def eval(self, data:Dict[str, List[InferenceInput]], eval_configs) -> Dict[str, List[InferenceOutput]]:
        task2details = {}
        for task, input in data.items():
            raw_output = self.generation(input)
            for item in raw_output:
                item.prompt = re.sub(r"<image>", "", item.prompt)
                item.raw_output.prompt = re.sub(r"<image>", "", item.raw_output.prompt)
            task2details[task] = raw_output
        
        return task2details
    
    def _generation(self, inputs: List[InferenceInput]) -> List[InferenceOutput]:
        assert isinstance(inputs, list)
        InferenceOutputs = []
        outputs = self.model.generate([{"prompt": input.text, "multi_modal_data": {"image": input.image_file},} for input in inputs], sampling_params=self.samplingparams)
        InferenceOutputs = [InferenceOutput.from_vllm_output(question_id=input.question_id, vllm_output=output, store_raw=True) for output, input in zip(outputs, inputs)]
        return InferenceOutputs

def evaluator(test_dataset, output_data, file_path, api_key, base_url):
    num_match = 0
    num_sum = 0
    question_id = set()
    for test_item in tqdm(test_dataset, desc="Evaluating"):
        for output_item in output_data:
            if test_item['index'] == output_item.question_id and output_item.question_id not in question_id:
                question_id.add(output_item.question_id)
                num_sum += 1
                choices = '(A) ' + test_item["A"] + '\n(B) ' + test_item["B"] + '\n(C) ' + test_item["C"] + '\n(D) ' + test_item["D"]
                true_or_false = judger(test_item['question'], choices, test_item['answer'], output_item.response[0], api_key, base_url)
                if true_or_false:
                    num_match += 1
                save_detail(test_item['question'], output_item.prompt, test_item['answer'], output_item.response[0], true_or_false, file_path)

    return num_match, num_sum

def judger(question, choices, correct_answer, response, api_key, base_url):
    if correct_answer not in response:
        return False
    match = re.search(r'(?<![a-zA-Z])[A-Z](?![a-zA-Z])', response)
    if match:
        return correct_answer == match.group()
    return gpt_judge_answer(question, choices, response, api_key, base_url)

def gpt_judge_answer(question, choices, response, api_key, base_url):
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
        INSERT_CHOICES_HERE=choices,
        INSERT_TEXT_OF_RESPONSE_HERE=response
    )

    choice = get_response(prompt).strip()
    match = re.search(r'(?<![a-zA-Z])[A-DZ](?![a-zA-Z])', choice)
    if match:
        return match.group()
    else:
        return "Z"


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    _, unparsed_args = parser.parse_known_args()
    keys = [k[2:] for k in unparsed_args[0::2]]
    values = list(unparsed_args[1::2])
    unparsed_args = dict(zip(keys, values))

    dict_configs, infer_configs = read_eval_cfgs('mmbench', 'vLLM')

    try:
        assert dict_configs or infer_configs, "Config file does not exist or is incomplete."
    except AssertionError as e:
        print("Config file is not exist or incomplete.")
        exit()
    
    for k, v in unparsed_args.items():
        if v == '' or v is None:
            continue
        dict_configs = update_dict(dict_configs, custom_cfgs_to_dict(k, v))
        infer_configs = update_dict(infer_configs, custom_cfgs_to_dict(k, v))
    
    dict_configs, infer_configs = dict_to_namedtuple(dict_configs), dict_to_namedtuple(infer_configs)
    model_config = dict_configs.default.model_cfgs
    data_cfgs = dict_configs.default.data_cfgs
    eval_configs = dict_configs.default.eval_cfgs
    logger = EvalLogger('Evaluation', log_dir=eval_configs.output_dir)
    dataloader = MMBenchDataLoader(dict_configs)
    assert not (dataloader.num_shot > 0 or dataloader.cot), "Few-shot or chain-of-thought cannot be used for this benchmark."
    test_data = dataloader.load_dataset()
    eval_module = MMBenchGeneratorVLLM(model_config, infer_configs)
    raw_outputs_dir = os.path.join(eval_configs.output_dir, f"raw_outputs_{re.sub(r'/', '_', model_config.model_name_or_path)}.pkl")
    if os.path.exists(raw_outputs_dir):
        raw_outputs = load_raw_outputs(raw_outputs_dir)
    else:
        raw_outputs = eval_module.eval(test_data, eval_configs)
        save_raw_outputs(raw_outputs, raw_outputs_dir)

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
    for task, _ in raw_outputs.items():
        test_data = load_dataset(data_cfgs.task_dir, task)[data_cfgs.split]
        file_path = f"{uuid_path}/{task}.json"
        num_match, num_sum = evaluator(test_data, raw_outputs[task], file_path, api_key, base_url)
        tot_num_match += num_match
        tot_num_sum += num_sum

        output_dict = {
            'model_id': [dict_configs.default.model_cfgs.model_id],
            'num_match': [num_match],
            'num_sum': [num_sum],
            'accuracy': [num_match*100 / num_sum]
        }
        logger.print_table(title=f'MMBench/{task} Benchmark', data=output_dict)
        logger.log('info', '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        logger.log('info', f"task: {task}")
        logger.log('info', f"model_id: {output_dict['model_id'][0]},")
        logger.log('info', f"num_match: {output_dict['num_match'][0]},")
        logger.log('info', f"num_sum: {output_dict['num_sum'][0]},")
        logger.log('info', f"accuracy: {output_dict['accuracy'][0]},")
        logger.log('info', '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

    output_dict = {
        'model_id': [dict_configs.default.model_cfgs.model_id],
        'tot_num_match': [tot_num_match],
        'tot_num_sum': [tot_num_sum],
        'tot_accuracy': [tot_num_match*100 / tot_num_sum]
    }
    logger.print_table(title=f'MMBench Benchmark', data=output_dict)
    logger.log('info', '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    logger.log('info', f"model_id: {output_dict['model_id'][0]},")
    logger.log('info', f"tot_num_match: {output_dict['tot_num_match'][0]},")
    logger.log('info', f"tot_num_sum: {output_dict['tot_num_sum'][0]},")
    logger.log('info', f"tot_accuracy: {output_dict['tot_accuracy'][0]},")
    logger.log('info', '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

if __name__ == '__main__':
    main()
