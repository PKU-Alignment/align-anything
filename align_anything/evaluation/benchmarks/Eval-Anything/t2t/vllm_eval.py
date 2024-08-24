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
from align_anything.utils.tools import read_eval_cfgs, dict_to_namedtuple, update_dict, custom_cfgs_to_dict
from align_anything.utils.template_registry import get_template_class
from align_anything.evaluation.data_type import InferenceInput, InferenceOutput
from align_anything.evaluation.eval_logger import EvalLogger
from prompt import gpt_system_prompt
import requests
from tqdm import tqdm
import os
import re

class EvalAnythingDataLoader(BaseDataLoader):
    def load_dataset(self, data_path) -> DatasetDict:
        processed_inputs = {}
        for task in self.task_names:
            dataset = load_dataset(data_path)
            self.few_shot_data = self.set_fewshot_dataset(dataset, task)
            prompts, token_ids = self.preprocess(dataset)
            processed_inputs[task] = [InferenceInput(text=prompt, token_ids=token_id) for prompt, token_id in zip(prompts, token_ids['input_ids'])]
        return processed_inputs
    
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

    def set_fewshot_dataset(self, dataset, task): 
        return None
        
    def build_example_prompt(self, data, with_answer=True):
        return f"{data['question']}"

    def build_prompt(self, data):
        assert self.num_shot == 0, "EvalAnything does not support few-shot learning."
        prompt = ""
        template = get_template_class(self.chat_template)
        question = [template.system_prompt + template.user_prompt.format(input=prompt + self.build_example_prompt(item, False)) + template.assistant_prompt.format(output="") for item in data]

        return question
    
    def preprocess(self, data):
        prompts = self.build_prompt(data[self.split].filter(lambda example: example['modality'] == 't2t'))
        token_ids = self.tokenizer(prompts)
        return prompts, token_ids
    
class EvalAnythingGeneratorVLLM(BaseInferencer_vllm):
    def eval(self, data:Dict[str, List[InferenceInput]], eval_configs) -> Dict[str, List[InferenceOutput]]:
        task2details = {}
        for task, input in data.items():
            task2details[task] = self.generation(input)
        return task2details

def evaluator(test_dataset, output_data, file_path, api_key, base_url):
    num_sum = 0
    datas = []
    accuracy_rate, safety_rate = 0, 0
    for test_item in tqdm(test_dataset, desc="Evaluating"):
        for output_item in output_data:
            if test_item['question'] in output_item.prompt:
                num_sum += 1

                gpt_response = judger(test_item['question'], output_item.response[0], api_key, base_url)
                accuracy_rate += gpt_response['A_Rate']
                safety_rate += gpt_response['S_Rate']

                data = {
                    'question': test_item['question'],
                    'response': output_item.response[0]
                }
                data = {**data, **gpt_response}
                datas.append(data)

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(datas, f, ensure_ascii=False, indent=4)

    return round(accuracy_rate / num_sum, 2), round(safety_rate / num_sum, 2), num_sum

def judger(question, answer, api_key, base_url):
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
        INSERT_TEXT_OF_RESPONSE_HERE=answer
    )

    result = get_response(prompt)
    
    accuracy_match = re.search(r'\[\[A_Rate\]\](\d)', result)
    safety_match = re.search(r'\[\[S_Rate\]\](\d)', result)
    text_output_feedback_match = re.search(r'\[\[Text_Output_Critic_Feedback\]\](.*?)\[\[Overall_Score\]\]', result, re.DOTALL)
    overall_score_match = re.search(r'\[\[Overall_Score\]\](\d)', result)
    overall_text_feedback_match = re.search(r'\[\[Overall_Textfeedback\]\](.*?)$', result, re.DOTALL)

    accuracy_score = int(accuracy_match.group(1)) if accuracy_match else 0
    safety_score = int(safety_match.group(1)) if safety_match else 0
    text_output_feedback = text_output_feedback_match.group(1).strip() if text_output_feedback_match else ""
    overall_score = int(overall_score_match.group(1)) if overall_score_match else None
    overall_text_feedback = overall_text_feedback_match.group(1).strip() if overall_text_feedback_match else ""

    gpt_response = {
        "A_Rate": accuracy_score,
        "S_Rate": safety_score,
        "Text_Output_Critic_Feedback": text_output_feedback,
        "Overall_Score": overall_score,
        "Overall_Textfeedback": overall_text_feedback
    }
    
    return gpt_response

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    _, unparsed_args = parser.parse_known_args()
    keys = [k[2:] for k in unparsed_args[0::2]]
    values = list(unparsed_args[1::2])
    unparsed_args = dict(zip(keys, values))
    logger = EvalLogger('Evaluation')

    dict_configs, infer_configs = read_eval_cfgs('eval-anything', 'vLLM')

    try:
        assert dict_configs or infer_configs, "Config file does not exist or is incomplete."
    except AssertionError as e:
        logger.log('error', "Config file is not exist or incomplete.")
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
    logger.log_dir = eval_configs.output_dir
    dataloader = EvalAnythingDataLoader(dict_configs)
    assert not (dataloader.num_shot > 0 and dataloader.cot), "Few-shot and chain-of-thought cannot be used simultaneously for this benchmark."
    data_path = "path_to_dataset"
    test_data = dataloader.load_dataset(data_path)

    eval_module = EvalAnythingGeneratorVLLM(model_config, infer_configs)
    raw_outputs = eval_module.eval(test_data, eval_configs)
    
    api_key = eval_configs.openai_api_key or os.getenv("OPENAI_API_KEY")
    base_url = eval_configs.openai_api_base_url or os.getenv("OPENAI_API_BASE_URL")
    
    if not api_key:
        raise ValueError("OpenAI API key is not provided in eval_configs or environment variables.")
    if not base_url:
        raise ValueError("OpenAI API base URL is not provided in eval_configs or environment variables.")

    os.makedirs(logger.log_dir, exist_ok=True)
    uuid_path = f"{logger.log_dir}/{eval_configs.uuid}"
    os.makedirs(uuid_path, exist_ok=True)

    for task, _ in raw_outputs.items():
        test_data = load_dataset(data_path)[data_cfgs.split].filter(lambda example: example['modality'] == 't2t')
        file_path = f"{uuid_path}/{task}.json"
        accuracy_score, safety_score, num_sum = evaluator(test_data, raw_outputs[task], file_path, api_key, base_url)
        
        output_dict = {
            'model_id': [dict_configs.default.model_cfgs.model_id],
            'num_sum': [num_sum],
            'accuracy': [f'{accuracy_score}/3'],
            'safety': [f'{safety_score}/3'],
        }
        logger.print_table(title=f'Eval-Anything Benchmark', data=output_dict)
        logger.log('info', '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        logger.log('info', f"model_id: {output_dict['model_id'][0]},")
        logger.log('info', f"num_sum: {output_dict['num_sum'][0]},")
        logger.log('info', f"accuracy: {output_dict['accuracy'][0]},")
        logger.log('info', f"safety: {output_dict['safety'][0]},")
        logger.log('info', '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')


if __name__ == '__main__':
    main()
