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
from datasets import load_dataset
from align_anything.utils.tools import read_eval_cfgs, dict_to_namedtuple, update_dict, custom_cfgs_to_dict, save_raw_outputs, load_raw_outputs
from align_anything.utils.template_registry import get_template_class
from align_anything.evaluation.data_type import InferenceInput, InferenceOutput
import requests
import re
from align_anything.evaluation.eval_logger import EvalLogger

class LongBenchDataLoader(BaseDataLoader):
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
        if self.cot:
            with open('../cot_fewshot/LongBench/longbench.json', 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data
        else:
            return None
    
    def build_example_prompt(self, data, with_answer=True, cot=False):
        answer = f'Answer: {self.get_answer(data)}' if with_answer else 'Answer: '
        return f"{data['context']}\n{data['input']}\n{answer}"

    def build_prompt(self, data):
        prompt = ""
        prompt = f"The following are questions (with stories).\n\n"
        cot_prompt = f"Let's think step by step. "
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

class LongBenchGeneratorVLLM(BaseInferencer_vllm):
    def eval(self, data:Dict[str, List[InferenceInput]], eval_configs) -> Dict[str, List[InferenceOutput]]:
        task2details = {}
        for task, input in data.items():
            task2details[task] = self.generation(input)      
        return task2details

def evaluator(raw_output: List[InferenceOutput], dataloader: LongBenchDataLoader, task: str, file_path, gpt_data, gpt_data_file, api_key, base_url):
    dataset = load_dataset(dataloader.task_dir, task)[dataloader.split]
    responses = []
    cnt_sum = 0
    for item in raw_output:
        responses.append(
            {
                'prompt': (item.prompt),
                'answer': item.response[0] if item.response else ""
            }
        )
    total_score = 0.0
    for correct_answer in tqdm(dataset, desc="Evaluating"):
        cnt_sum += 1
        for response in responses:
            if correct_answer['input'] in response['prompt']:
                gpt_id = correct_answer['input'] + ''.join(correct_answer['answers']) + response['answer']
                if gpt_id in gpt_data:
                    score = gpt_data[gpt_id]
                else:
                    score = judger(correct_answer['context'], correct_answer['input'], correct_answer['answers'], response['answer'], api_key, base_url)
                    gpt_data[gpt_id] = score
                total_score += score
                save_detail(correct_answer['context'] + '\n' + correct_answer['input'], '', correct_answer['answers'], response['answer'], score, file_path)
                break
        
    with open(gpt_data_file, 'w', encoding='utf-8') as f:
        json.dump(gpt_data, f, ensure_ascii=False, indent=4)

    return cnt_sum, total_score / cnt_sum

def gpt4_judger(text, question, answers, response, api_key, base_url):
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
    correct_answer = '\n'.join([f'{answers[idx]}' for idx in range(len(answers))])
    CRITERIA = {"accuracy": """
                Score 1: The Answer is completely unrelated to the reference.
                Score 3: The Answer has minor relevance but does not align with the reference.
                Score 5: The Answer has moderate relevance but contains inaccuracies.
                Score 7: The Answer aligns with the reference but has minor omissions.
                Score 10: The Answer is completely accurate and aligns perfectly with the reference.
                Only respond with a numberical score"""}
    prompt = f"\nQuestion: {question}\nReference: {correct_answer}\nAnswer: {response}\n{CRITERIA['accuracy']}"
    score = get_response(prompt)

    return score

def judger(text, question, answers, response, api_key, base_url):
    score = gpt4_judger(text, question, answers, response, api_key, base_url)
    if not isinstance(score, float):
        match = re.search(r'\d+(\.\d+)?', score)
        if match:
            score = float(match.group())
            if score > 10:
                score = 1
        else:
            score = 1

    return score
  
def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    _, unparsed_args = parser.parse_known_args()
    keys = [k[2:] for k in unparsed_args[0::2]]
    values = list(unparsed_args[1::2])
    unparsed_args = dict(zip(keys, values))
    
    dict_configs, infer_configs = read_eval_cfgs('longbench', 'vLLM')
    
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
    eval_configs = dict_configs.default.eval_cfgs
    logger = EvalLogger('Evaluation', log_dir=eval_configs.output_dir)
    dataloader = LongBenchDataLoader(dict_configs)
    test_data = dataloader.load_dataset()
    eval_module = LongBenchGeneratorVLLM(model_config, infer_configs)
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

    gpt_data_file = os.path.join(eval_configs.output_dir, f"gpt_data.json")
    gpt_data = {}
    if os.path.exists(gpt_data_file):
        with open(gpt_data_file, 'r', encoding='utf-8') as file:
            gpt_data = json.load(file)

    num_task, tot_num_sum, tot_avg_score = 0, 0, 0.0
    for task, _ in raw_outputs.items():

        file_path = f"{uuid_path}/{task}.json"
        cnt_sum, avg_score = evaluator(raw_outputs[task], dataloader, task, file_path, gpt_data, gpt_data_file, api_key, base_url)
        num_task += 1
        tot_num_sum += cnt_sum
        tot_avg_score += avg_score

        eval_results = {
            'model_id': [dict_configs.default.model_cfgs.model_id],
            'num_fewshot': [eval_configs.n_shot],
            'chain_of_thought': [eval_configs.cot],
            'num_sum': [cnt_sum],
            'average_score': [avg_score]
        }
        logger.print_table(title=f'LongBench/{task} Benchmark', data=eval_results)
        logger.log('info', '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        logger.log('info', f"task: {task}")
        logger.log('info', f"model_id: {eval_results['model_id'][0]},")
        logger.log('info', f"num_fewshot: {eval_results['num_fewshot'][0]},")
        logger.log('info', f"chain_of_thought: {eval_results['chain_of_thought'][0]},")
        logger.log('info', f"num_sum: {eval_results['num_sum'][0]},")
        logger.log('info', f"average_score: {eval_results['average_score'][0]},")
        logger.log('info', '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

    eval_results = {
        'model_id': [dict_configs.default.model_cfgs.model_id],
        'num_fewshot': [eval_configs.n_shot],
        'chain_of_thought': [eval_configs.cot],
        'tot_num_sum': [tot_num_sum],
        'tot_average_score': [tot_avg_score / num_task]
    }
    logger.print_table(title=f'LEval Benchmark', data=eval_results)
    logger.log('info', '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    logger.log('info', f"model_id: {eval_results['model_id'][0]},")
    logger.log('info', f"num_fewshot: {eval_results['num_fewshot'][0]},")
    logger.log('info', f"chain_of_thought: {eval_results['chain_of_thought'][0]},")
    logger.log('info', f"tot_num_sum: {eval_results['tot_num_sum'][0]},")
    logger.log('info', f"tot_average_score: {eval_results['tot_average_score'][0]},")
    logger.log('info', '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

if __name__ == '__main__':
    main()
