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
import argparse
from align_anything.evaluation.inference.vllm_inference import BaseInferencer_vllm
from align_anything.evaluation.dataloader.base_dataloader import BaseDataLoader
from typing import List, Dict
from align_anything.utils.tools import read_eval_cfgs, dict_to_namedtuple, update_dict, custom_cfgs_to_dict
from datasets import load_dataset, DatasetDict
from align_anything.utils.template_registry import get_template_class
from align_anything.evaluation.data_type import InferenceInput, InferenceOutput
from align_anything.evaluation.inference.vllm_inference import update_results
from align_anything.evaluation.eval.base_eval import API_Single_Eval
from align_anything.evaluation.eval_logger import EvalLogger
import re
import json

class MTBenchDataLoader(BaseDataLoader):
    def get_task_names(self):
        if isinstance(self.data_cfgs.task, list):
            return self.data_cfgs.task
        else:
            task_names = [
            self.data_cfgs.task
            ]
            return task_names
        
    def load_dataset(self) -> DatasetDict:
        processed_inputs = {}
        for task in self.task_names:
            current_file_path = os.path.abspath(__file__)
            current_dir = os.path.dirname(current_file_path)
            dataset = load_dataset(os.path.join(current_dir, self.data_cfgs.task_dir))
            prompts, token_ids = self.preprocess(dataset)
            processed_inputs[task] = [InferenceInput(text=prompt, token_ids=token_id) for prompt, token_id in zip(prompts, token_ids['input_ids'])]
        return processed_inputs
    
    def build_prompt(self, data, responses_r1=None):
        cot_prompt = f"Let's think step by step."
        template = get_template_class(self.chat_template)

        if self.cot:
            if responses_r1:
                return [template.system_prompt + \
                    template.user_prompt.format(input=item['instruction'][0]) + \
                    template.assistant_prompt.format(output=response_r1) + \
                    template.user_prompt.format(input=item['instruction'][1]) + \
                    template.assistant_prompt.format(output=cot_prompt) \
                    for response_r1, item in zip(responses_r1, data)]
            else:
                return [template.system_prompt + template.user_prompt.format(input=item['instruction'][0]) + template.assistant_prompt.format(output=cot_prompt) for item in data]

        else:
            if responses_r1:
                return [template.system_prompt + \
                    template.user_prompt.format(input=item['instruction'][0]) + \
                    template.assistant_prompt.format(output=response_r1) + \
                    template.user_prompt.format(input=item['instruction'][1]) + \
                    template.assistant_prompt.format(output="") \
                    for response_r1, item in zip(responses_r1, data)]
            else:
                return [template.system_prompt + template.user_prompt.format(input=item['instruction'][0]) + template.assistant_prompt.format(output="") for item in data]
    
    def load_dataset_round2(self, outputs_r1: Dict[str, List[InferenceOutput]]):
        processed_inputs = {}
        for task in self.task_names:
            current_file_path = os.path.abspath(__file__)
            current_dir = os.path.dirname(current_file_path)
            dataset = load_dataset(os.path.join(current_dir, self.data_cfgs.task_dir))
            responses_r1 = [output_r1.response[0] for output_r1 in outputs_r1[task]]
            prompts, token_ids = self.preprocess(data=dataset, responses_r1=responses_r1)
            processed_inputs[task] = [InferenceInput(text=prompt, token_ids=token_id) for prompt, token_id in zip(prompts, token_ids['input_ids'])]
        return processed_inputs
    
    def preprocess(self, data, responses_r1=None):
        prompts = self.build_prompt(data[self.split], responses_r1)
        token_ids = self.tokenizer(prompts)
        return prompts, token_ids

class MTBenchGeneratorVLLM(BaseInferencer_vllm):
    def eval(self, data:Dict[str, List[InferenceInput]], eval_configs) -> Dict[str, List[InferenceOutput]]:
        task2details = {}
        for task, input in data.items():
            task2details[task] = self.generation(input)
        
        output_dir = eval_configs.output_dir
        brief_filename = eval_configs.brief_filename
        model_id = self.model_cfgs.model_id
        detailed_filename = f'{model_id}_detailed'
        brief_filename = f'{model_id}_brief'
        update_results(output_dir, brief_filename, detailed_filename,task2details)
        
        return task2details

def fill_prompt_template(prompt_template, **kwargs):
    return prompt_template.format(**kwargs)

def get_score(response: str):
    pattern = r'\[\[(-?\d+(?:\.\d+)?)\]\]'
    match = re.search(pattern, response)
    if match:
        return match.group(1)
    else:
        return None

class API_Eval(API_Single_Eval):
    def build_gpt_input(self, judge_prompt: str, user_prompt : str):
        input = [{'role': 'system', 'content': judge_prompt},
                {'role': 'user', 'content': user_prompt}]
    
        return input


def evaluator(raw_output1: List[InferenceOutput], raw_output2: List[InferenceOutput], dataloader: MTBenchDataLoader, task: str, eval_configs= None):
    current_file_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file_path)
    dataset = load_dataset(current_dir,task)[dataloader.split]
    dataset = load_dataset(current_dir,split='train',data_files='test.jsonl')
    prompts= []
    file_path = "./judge_prompts.jsonl"
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            prompt = json.loads(line.strip())
            prompts.append(prompt)
    
    questions=[]
    raw_responses = []
    responses = []
    eval_case = []
    for instance in dataset :
        questions.append(
            {
                'question_id': instance['id'],
                'category': instance['description'],
                'turns': instance['instruction'],
                'reference': instance['reference']
            }
        )
    
    for item1,item2 in zip(raw_output1,raw_output2) :
        raw_responses.append(
            {
                'prompt': item2.prompt,
                'response_1': item1.response,
                'response_2': item2.response
            }
        )
    for question, raw_response in zip(questions,raw_responses):
        question_turns = question['turns']
        question_1 = question_turns[0]
        question_2 = question_turns[1]
        reference_turns = question['reference']
        if not reference_turns:
            ref_answer_1 = ""
            ref_answer_2 = ""
        else:
            ref_answer_1 = reference_turns[0]
            ref_answer_2 = reference_turns[1]
        
        
        answer_1 = raw_response['response_1'][0]
        answer_2 = raw_response['response_2'][0]
        responses.append(
            {
                'question_id': question['question_id'],
                'category': question['category'],
                'question_1': question_1,
                'question_2': question_2,
                'ref_answer_1':ref_answer_1,
                'ref_answer_2': ref_answer_2,
                'answer_1': answer_1,
                'answer_2' : answer_2
            }
        )
    system_prompts = []
    user_prompts = []
    judger = API_Eval(model = eval_configs.judge_model, num_workers = 20, temperature = 0, template_function= None, api_key=eval_configs.openai_api_key, base_url=eval_configs.openai_api_base_url )
    for response in responses:
        
        if not response['ref_answer_1']:
             judge_prompt= prompts[6]['system_prompt']
             prompt_template = prompts[6]['prompt_template']
             user_prompt = fill_prompt_template(prompt_template, question_1 = response['question_1'], question_2 = response['question_2'], answer_1 = response['answer_1'], answer_2 = response['answer_2'])
             system_prompts.append(judge_prompt)
             user_prompts.append(user_prompt)
        else: 
            judge_prompt= prompts[7]['system_prompt']
            prompt_template = prompts[7]['prompt_template']
            user_prompt = fill_prompt_template(prompt_template, question_1 = response['question_1'], question_2 = response['question_2'], ref_answer_1=response['ref_answer_1'], ref_answer_2 = response['ref_answer_1'], answer_1 = response['answer_1'], answer_2 = response['answer_2'])
            system_prompts.append(judge_prompt)
            user_prompts.append(user_prompt)
    
    results = judger.evaluate(system_prompts, user_prompts)
    for response, system_prompt, user_prompt, result in zip(responses, system_prompts, user_prompts, results):
            output = result.raw_output.choices[0].message.content
            score = get_score(output)
            time = 0
            while score is None:
                multi_results=[]
                multi_results = judger.evaluate(system_prompts=[system_prompt],user_prompts=[user_prompt])
                output = multi_results[0].raw_output.choices[0].message.content
                score = get_score(output)
                time+=1
                if time >=10:
                    score = 0
                    break
            
            eval_case.append(
                 {
                    'question_id' : response['question_id'],
                    'system_prompt': system_prompt,
                    'user_prompt' : user_prompt,
                    'response': output,
                    'score': score
                 }
             )       
    
    return responses, eval_case

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    _, unparsed_args = parser.parse_known_args()
    keys = [k[2:] for k in unparsed_args[0::2]]
    values = list(unparsed_args[1::2])
    unparsed_args = dict(zip(keys, values))
    logger = EvalLogger('Evaluation')
    
    dict_configs, infer_configs = read_eval_cfgs('mt_bench', 'vLLM')

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
    eval_configs = dict_configs.default.eval_cfgs

    dataloader = MTBenchDataLoader(dict_configs)
    assert not (dataloader.num_shot > 0 and dataloader.cot), "Few-shot and chain-of-thought cannot be used simultaneously for this benchmark."
    test_data_round1 = dataloader.load_dataset()
    eval_module = MTBenchGeneratorVLLM(model_config, infer_configs)
    output_data_round1 = eval_module.eval(test_data_round1, eval_configs)
    
    test_data_round2 = dataloader.load_dataset_round2(output_data_round1)
    output_data_round2 = eval_module.eval(test_data_round2, eval_configs)
    file_name = f'{eval_module.model_id}'+f'_cot_{dict_configs.default.eval_cfgs.cot}'
    responses=[]
    evals=[]
    for task, _ in output_data_round2.items():
        responses,evals = evaluator(output_data_round1[task],output_data_round2[task], dataloader, task, eval_configs)
    
    merged_list=[]
    for resp, eval_ in zip(responses, evals):
        merged_dict = {**resp, **eval_}
        merged_list.append(merged_dict)

    os.makedirs(eval_configs.output_dir,exist_ok=True)
    raw_result_file = os.path.join(eval_configs.output_dir,file_name + "_raw_result.jsonl")
    with open(raw_result_file, 'w') as file:
        for item in merged_list:
            json.dump(item, file)
            file.write('\n')

    category_list=[]

    total_score = 0
    total_count = 0
    for item in merged_list:
        category = item['category']
        score = float(item['score'])
        
        found = False
        for cat_dict in category_list:
            if cat_dict['category'] == category:
                cat_dict['total_score'] += score
                cat_dict['count'] += 1
                found = True
                break
        if not found:
            category_list.append(
                {
                    'category': category,
                    'total_score': score,
                    'count': int(1),
                    'average_score': float(0),
                    'example': item
                }
            )
        total_score+=score
        total_count+=1
        
    for cate in category_list:
        cate['average_score'] = float(cate['total_score'])/(cate['count'])
    for cate in category_list:
        logger.log('info', '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        logger.log('info', f'Category: {cate["category"]}')
        logger.log('info', f'Average Score: {cate["average_score"]}')
        logger.log('info', f'Num of Question: {cate["count"]}')
        logger.log('info', '===============================EXAMPLE===============================')
        logger.log('info', f'Question_1: {cate["example"]["question_1"]}')
        logger.log('info', f'Answer_1: {cate["example"]["answer_1"]}')
        if not cate["example"]["ref_answer_1"]:
            logger.log('info', f'Reference_1: {cate["example"]["ref_answer_1"]}')
        logger.log('info', f'Question_2: {cate["example"]["question_2"]}')
        logger.log('info', f'Answer_2: {cate["example"]["answer_2"]}')
        if not cate["example"]["ref_answer_2"]:
            logger.log('info', f'Reference_2: {cate["example"]["ref_answer_2"]}')
        logger.log('info', f'Score: {cate["example"]["score"]}')
        logger.log('info', f'Reasoning: {cate["example"]["response"]}')
    
    total_average = float(total_score)/float(total_count)
    logger.log('info', '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    logger.log('info', '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    logger.log('info', f'Average Score: {total_average}Total question: {total_count}')
     
    eval_results = {
            'total average': [float(total_average)],
            'total question': [total_count]
            }
    logger.print_table(title="Evaluation Results", data=eval_results)

if __name__ == '__main__':
    main()
