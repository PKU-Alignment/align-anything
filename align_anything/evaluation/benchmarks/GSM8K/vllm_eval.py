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
import re
from align_anything.evaluation.eval_logger import EvalLogger

class GSM8KDataLoader(BaseDataLoader):
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
            with open('../cot_fewshot/GSM8K/gsm8k.json', 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data
        else:
            return dataset['train']
    
    def build_example_prompt(self, data, with_answer=True, cot=False):
        answer = f'Answer: {self.get_answer(data)}' if with_answer else 'Answer: '
        return f"{data['question']}\n{answer}"

    def build_prompt(self, data):
        prompt = "The following are math word problems (with answers). Please provide the final answer on a new line after '####'. For example: #### 42\n\n"
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

class GSM8KGeneratorVLLM(BaseInferencer_vllm):
    def eval(self, data:Dict[str, List[InferenceInput]], eval_configs) -> Dict[str, List[InferenceOutput]]:
        task2details = {}
        for task, input in data.items():
            task2details[task] = self.generation(input)      
        return task2details

def evaluator(raw_output: List[InferenceOutput], dataloader: GSM8KDataLoader, task: str, file_path):
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
                'question': instance['question'],
                'answer': dataloader.get_answer(instance)
            }
        )
    for item in raw_output:
        responses.append(
            {
                'prompt': (item.prompt),
                'generated_answer': item.response[0] if item.response else ""
            }
        )
    for correct_answer in tqdm(correct_answers, desc="Evaluating"):
        cnt_sum += 1
        for response in responses:
            if correct_answer['question'] in response['prompt']:
                flag_fail = False
                answer = correct_answer['answer'].split('####')[-1].strip()
                generated_answer = get_generated_answer(response['generated_answer'])
                if generated_answer == answer:
                    cnt_match += 1
                save_detail(correct_answer['question'], '', correct_answer['answer'], response['generated_answer'], generated_answer == answer, file_path)
                break
        if flag_fail:
            cnt_fail += 1
        else:
            flag_fail = True
        
    return cnt_match, cnt_sum

def get_last_number(data):
    data = data.replace(',', '')
    numbers = re.findall(r'\d+', data)
    if numbers:
        return numbers[-1]
    else:
        return ''

def get_generated_answer(data):
    answer = ''
    if '####' in data:
        answer = get_last_number(data.split('####')[-1].strip())
    if not answer:
        answer = get_last_number(data)
    return answer
    
def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    _, unparsed_args = parser.parse_known_args()
    keys = [k[2:] for k in unparsed_args[0::2]]
    values = list(unparsed_args[1::2])
    unparsed_args = dict(zip(keys, values))
    
    dict_configs, infer_configs = read_eval_cfgs('gsm8k', 'vLLM')
    
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
    dataloader = GSM8KDataLoader(dict_configs)
    test_data = dataloader.load_dataset()
    eval_module = GSM8KGeneratorVLLM(model_config, infer_configs)
    raw_outputs_dir = os.path.join(eval_configs.output_dir, f"raw_outputs_{re.sub(r'/', '_', model_config.model_name_or_path)}.pkl")
    if os.path.exists(raw_outputs_dir):
        raw_outputs = load_raw_outputs(raw_outputs_dir)
    else:
        raw_outputs = eval_module.eval(test_data, eval_configs)
        save_raw_outputs(raw_outputs, raw_outputs_dir)
    
    os.makedirs(logger.log_dir, exist_ok=True)
    uuid_path = f"{logger.log_dir}/{eval_configs.uuid}"
    os.makedirs(uuid_path, exist_ok=True)

    for task, _ in raw_outputs.items():

        file_path = f"{uuid_path}/{task}.json"
        cnt_match, cnt_sum = evaluator(raw_outputs[task], dataloader, task, file_path)

        eval_results = {
            'model_id': [dict_configs.default.model_cfgs.model_id],
            'num_fewshot': [eval_configs.n_shot],
            'chain_of_thought': [eval_configs.cot],
            'num_match': [cnt_match],
            'num_sum': [cnt_sum],
            'accuracy': [cnt_match / cnt_sum]
        }
        logger.print_table(title=f'GSM8K/{task} Benchmark', data=eval_results)
        logger.log('info', '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        logger.log('info', f"task: {task}")
        logger.log('info', f"model_id: {eval_results['model_id'][0]},")
        logger.log('info', f"num_fewshot: {eval_results['num_fewshot'][0]},")
        logger.log('info', f"chain_of_thought: {eval_results['chain_of_thought'][0]},")
        logger.log('info', f"num_match: {eval_results['num_match'][0]},")
        logger.log('info', f"num_sum: {eval_results['num_sum'][0]},")
        logger.log('info', f"accuracy: {eval_results['accuracy'][0]},")
        logger.log('info', '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

if __name__ == '__main__':
    main()
