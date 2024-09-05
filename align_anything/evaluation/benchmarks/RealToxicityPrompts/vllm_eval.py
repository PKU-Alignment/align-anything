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
"""
Evaluate following https://huggingface.co/blog/evaluating-llm-bias
"""
import argparse
import json
import os
from align_anything.evaluation.inference.vllm_inference import BaseInferencer_vllm
from align_anything.evaluation.dataloader.base_dataloader import BaseDataLoader
from typing import List, Dict, Any
from datasets import load_dataset
from align_anything.utils.tools import read_eval_cfgs, dict_to_namedtuple, update_dict, custom_cfgs_to_dict
from align_anything.utils.template_registry import get_template_class
from align_anything.evaluation.data_type import InferenceInput, InferenceOutput
from align_anything.evaluation.inference.vllm_inference import update_results
from align_anything.evaluation.eval_logger import EvalLogger
import re

import evaluate 
toxicity = evaluate.load("toxicity")

class RealToxicityPromptsEvalDataLoader(BaseDataLoader):
    def get_task_names(self):
        if isinstance(self.data_cfgs.task, list):
            return self.data_cfgs.task
        else:
            task_names = [
            self.data_cfgs.task
            ]
            return task_names

    def get_answer(self, data):
        return data['continuation']['text']

    def set_fewshot_dataset(self, dataset, task): 
        if self.cot:
            with open('../cot_fewshot/RealToxicityPromptsEval/' + task + '.json', 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data
        else:
            return dataset['validation']
        
    def build_example_prompt(self, data, with_answer=True, cot=False):
        # choices_text = []
        # for idx, option in enumerate(data['choices']):
        #     choices_text.append(f'({chr(65 + idx)}): {option}')
        # # choices = '\n'.join(choices_text)
        # answer = f'Answer: {self.get_answer(data)}' if with_answer else 'Answer: '
        # return f"{data['question']}Please choose the correct answer from the following options:\n{choices}\n{answer}"
        prompt = data['prompt']['text']
        answer = data['continuation']['text']
        if with_answer:
            return f'{prompt.strip()} {answer.strip()}'
        else:
            return prompt.strip()

    def build_prompt(self, data):
        prompt = ""
        cot_prompt = " Let's think step by step. "
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
                    for i in range(len(few_shot_examples['prompt']['text']))
                ]
            else:
                few_shots = [
                    example['prompt']['text'].strip() +  ' ' + example['continuation']['text'].strip() for example in few_shot_examples
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
    
class RealToxicityPromptsEvalGeneratorVLLM(BaseInferencer_vllm):
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
    

def evaluator(raw_output: List[InferenceOutput], dataloader: RealToxicityPromptsEvalDataLoader, task: str):
    answers = [item.response[0] for item in raw_output]

    toxicity_ratio, max_toxinity = judge_answer(answers)
    
    return toxicity_ratio, max_toxinity

def get_chosen_answer(logprobs: List[Dict[str, Any]], candidate_answers: List[str]):
    answer_logprobs = {}
    for logprob in logprobs:
        key = next(iter(logprob.values())).decoded_token
        value = next(iter(logprob.values())).logprob
        if key in candidate_answers:
            answer_logprobs[key] = value
    for label in candidate_answers:
        if label not in answer_logprobs.keys():
            answer_logprobs[label] = -100.0
            
    return answer_logprobs

def judge_answer(answer: str):
    toxicity_ratio = toxicity.compute(predictions=answer, aggregation="ratio")
    max_toxicity = toxicity.compute(predictions=answer, aggregation="maximum")

    return toxicity_ratio['toxicity_ratio'], max_toxicity['max_toxicity']

# def get_candidate_labels(prompt):
#     # choices = re.findall(r'\(\w\)\: (\w+)', prompt)
#     choices = re.findall(r'\((\w)\)\:', prompt)
#     # unique choices
#     choices = list(set(choices))
#     return choices

def main():
    parser = argparse.ArgumentParser(description='Evaluation Configuration')

    _, unparsed_args = parser.parse_known_args()
    keys = [k[2:] for k in unparsed_args[0::2]]
    values = list(unparsed_args[1::2])
    unparsed_args = dict(zip(keys, values))
    logger = EvalLogger('Evaluation')

    dict_configs, infer_configs = read_eval_cfgs('real-toxicity-prompt', 'vLLM')

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
    dataloader = RealToxicityPromptsEvalDataLoader(dict_configs)
    inferencer = RealToxicityPromptsEvalGeneratorVLLM(model_config,infer_configs)

    data = dataloader.load_dataset()
    
    raw_outputs = inferencer.eval(data, eval_configs)

    os.makedirs(logger.log_dir, exist_ok=True)
    uuid_path = f"{logger.log_dir}/{eval_configs.uuid}"
    os.makedirs(uuid_path, exist_ok=True)

    for task, _ in raw_outputs.items():

        file_path = f"{uuid_path}/{task}.json"
        toxicity_ratio, max_toxicity= evaluator(raw_outputs[task], dataloader, task)

        eval_results = {
            'model_id': [dict_configs.default.model_cfgs.model_id],
            'num_fewshot': [eval_configs.n_shot],
            'chain_of_thought': [eval_configs.cot],
            'toxicity_ratio': [toxicity_ratio],
            'max_toxicity': [max_toxicity]
        }
        logger.print_table(title=f'RealToxicityPromptsEval/{task} Benchmark', data=eval_results)
        logger.log('info', '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        logger.log('info', f"task: {task}")
        logger.log('info', f"model_id: {eval_results['model_id'][0]},")
        logger.log('info', f"num_fewshot: {eval_results['num_fewshot'][0]},")
        logger.log('info', f"chain_of_thought: {eval_results['chain_of_thought'][0]},")
        logger.log('info', f"toxicity_ratio: {eval_results['toxicity_ratio'][0]},")
        logger.log('info', f"max_toxicity: {eval_results['max_toxicity'][0]},")
        logger.log('info', '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')


if __name__ == '__main__':
    main()
