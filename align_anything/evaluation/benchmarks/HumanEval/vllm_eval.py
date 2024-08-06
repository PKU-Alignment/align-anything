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
from align_anything.evaluation.inference.base_inference import BaseInferencer_vllm
from align_anything.evaluation.dataloader.base_dataloader import BaseDataLoader
from typing import List, Dict
from datasets import load_dataset
from align_anything.utils.tools import read_eval_cfgs, dict_to_namedtuple, update_dict, custom_cfgs_to_dict
from align_anything.utils.template_registry import get_template_class
from align_anything.evaluation.data_type import InferenceInput, InferenceOutput
from align_anything.evaluation.inference.base_inference import update_results
from align_anything.evaluation.eval_logger import EvalLogger
from datasets import Dataset
import multiprocessing
import torch
class HumanEvalDataLoader(BaseDataLoader):

    def get_task_names(self):
        if isinstance(self.data_cfgs.task, list):
            return self.data_cfgs.task
        else:
            task_names = [
            self.data_cfgs.task
            ]
            return task_names

    def get_answer(self, data):
        return data['canonical_solution']
    
    def get_fewshot_data(self):
        few_shot_examples = json.load(open("../few_shot.json", encoding='utf-8'))['humaneval']['ocp']

        formatted_data = []
        for example in few_shot_examples:
            formatted_data.append({
                'task_id': example['task_id'],
                'prompt': example['prompt'],
                'canonical_solution': example['canonical_solution'],
                'test': example['test'],
                'entry_point': example['entry_point']
            })

        return Dataset.from_dict({
            'task_id': [item['task_id'] for item in formatted_data],
            'prompt': [item['prompt'] for item in formatted_data],
            'canonical_solution': [item['canonical_solution'] for item in formatted_data],
            'test': [item['test'] for item in formatted_data],
            'entry_point': [item['entry_point'] for item in formatted_data]
        })

    def set_fewshot_dataset(self, dataset, task): 
        if self.cot:
            with open('../cot_fewshot/HumanEval/' + task + '.json', 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data
        else:
            return self.get_fewshot_data()
        
    def build_example_prompt(self, data, with_answer=True, cot=False):
        answer = f'Canonical_solution: {self.get_answer(data)}' if with_answer else 'Canonical_solution: '
        return f"Function description: {data['prompt']}\n{answer}"

    def build_prompt(self, data):
        prompt = f"The following are function description (with Canonical_solution).\n\n"
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
                    for i in range(len(few_shot_examples['prompt']))
                ]
            else:
                few_shots = [
                    f"{example['prompt']}\n'Canonical_solution: '{example['canonical_solution']}" for example in few_shot_examples
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
    
class HumanEvalGeneratorVLLM(BaseInferencer_vllm):

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

def evaluator(raw_output: List[InferenceOutput], dataloader: HumanEvalDataLoader, task: str):
    dataset = load_dataset(dataloader.task_dir, task)[dataloader.split]
    correct_answers = []
    responses = []
    true_cases = []
    false_cases = []
    cnt_sum = 0
    cnt_match = 0
    cnt_fail = 0
    flag_fail = True
    for instance in dataset:
        correct_answers.append(
            {
                'prompt': instance['prompt'],
                'test_data': instance['test'],
                'entry_point': instance['entry_point'],
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
    for correct_answer in correct_answers:
        cnt_sum += 1
        for response in responses:
            if correct_answer['prompt'] in response['prompt']:
                flag_fail = False
                generated_answer = get_generated_answer(response)
                eval_case = {
                    'prompt': correct_answer['prompt'],
                    'test_data': correct_answer['test_data'],
                    'correct_answer': correct_answer['answer'],
                    'generated_answer': generated_answer
                }
                if judge_answer(correct_answer, generated_answer):
                    cnt_match += 1
                    eval_case['result'] = True
                    true_cases.append(eval_case)
                else:
                    eval_case['result'] = False
                    false_cases.append(eval_case)
                break
        if flag_fail:
            cnt_fail += 1
        else:
            flag_fail = True
        
    return cnt_match, cnt_sum, true_cases, false_cases

def get_generated_answer(response):
    prefix_1 = '```python'
    len_prefix_1 = len(prefix_1)
    prefix_2 = '```'
    index_head = response['generated_answer'].find(prefix_1)
    index_tail = response['generated_answer'][index_head + len_prefix_1:].find(prefix_2)
    return response['generated_answer'][index_head + len_prefix_1:][:index_tail]

def model_eval(text, model, tokenizer, device):
    model.eval()
    model.to(device)

    inputs = tokenizer(text, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            repetition_penalty=1.2,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def code_test(code: str, time_limit: int) -> str:
    def worker(result_queue: multiprocessing.Queue):
        try:
            exec_globals = {}
            exec(code, exec_globals)
            result_queue.put(1)
        except Exception as e:
            result_queue.put(0)

    result_queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=worker, args=(result_queue,))
    process.start()

    process.join(time_limit)
    if process.is_alive():
        process.terminate()
        process.join()
        result_queue.put(0)
    return result_queue.get()

def judge_answer(data, response):
    check_program = (
        data["prompt"] + response + "\n" +
        data["test_data"] + "\n" +
        f"check({data['entry_point']})"
    )
    time_limit = 5
    return code_test(check_program, time_limit=time_limit)

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    _, unparsed_args = parser.parse_known_args()
    keys = [k[2:] for k in unparsed_args[0::2]]
    values = list(unparsed_args[1::2])
    unparsed_args = dict(zip(keys, values))
    logger = EvalLogger('Evaluation')
    
    dict_configs, infer_configs = read_eval_cfgs('humaneval', 'vLLM')

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
    dataloader = HumanEvalDataLoader(dict_configs)
    assert not (dataloader.num_shot > 0 and dataloader.cot), "Few-shot and chain-of-thought cannot be used simultaneously for this benchmark."
    test_data = dataloader.load_dataset()
    eval_module = HumanEvalGeneratorVLLM(model_config, infer_configs)
    raw_outputs = eval_module.eval(test_data, eval_configs)

    tasks, num_matches, num_instances, acc = [], [], [], []
    for task, _ in raw_outputs.items():

        cnt_match, cnt_sum, true_cases, false_cases = evaluator(raw_outputs[task], dataloader, task)

        tasks.append(task)
        num_matches.append(cnt_match)
        num_instances.append(cnt_sum)
        acc.append(cnt_match / cnt_sum)

        logger.log('info', '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        logger.log('info', f"task: {task}")
        logger.log('info', '==============================TRUE CASE==============================')
        if true_cases:
            logger.log('info', f'Prompt: {true_cases[0]["prompt"]}')
            logger.log('info', f'Test data: {true_cases[0]["test_data"]}')
            logger.log('info', f'Correct Answer: {true_cases[0]["correct_answer"]}')
            logger.log('info', f'ChoGeneratedsen Answer: {true_cases[0]["generated_answer"]}')
        logger.log('info', '==============================FALSE CASE==============================')
        if false_cases:
            logger.log('info', f'Prompt: {false_cases[0]["prompt"]}')
            logger.log('info', f'Test data: {false_cases[0]["test_data"]}')
            logger.log('info', f'Correct Answer: {false_cases[0]["correct_answer"]}')
            logger.log('info', f'Generated Answer: {false_cases[0]["generated_answer"]}')
        logger.log('info', '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

    eval_results = {
            'task': tasks,
            'num_fewshot': [eval_configs.n_shot] * len(tasks),
            'chain_of_thought': [eval_configs.cot] * len(tasks),
            'num_match': num_matches,
            'num_sum': num_instances,
            'acc': acc
            }
    logger.print_table(title="Evaluation Results", data = eval_results)

if __name__ == '__main__':
    main()
