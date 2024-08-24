import os
import pickle
import argparse
import json
from align_anything.evaluation.dataloader.base_dataloader import BaseDataLoader
from typing import List, Dict, Any
from datasets import load_dataset
from align_anything.utils.tools import read_eval_cfgs, dict_to_namedtuple, update_dict, custom_cfgs_to_dict
from align_anything.utils.template_registry import get_template_class
from align_anything.evaluation.data_type import InferenceOutput
from align_anything.evaluation.eval_logger import EvalLogger

class HybridQADataLoader(BaseDataLoader):
    def get_task_names(self):
        if isinstance(self.data_cfgs.task, list):
            return self.data_cfgs.task
        else:
            return [self.data_cfgs.task]

    def get_answer(self, data):
        return data['answer']

    def build_example_prompt(self, data, with_answer=True, cot=False):
        # Construct the question, context (table + text), and answer (if required)
        context = f"Table: {data['table']}\nText: {data['text']}"
        question = f"Question: {data['question']}"
        answer = f"Answer: {self.get_answer(data)}" if with_answer else "Answer: "
        return f"{context}\n{question}\n{answer}"

    def build_prompt(self, data):
        prompt = f"Answer the following questions using the provided table and text information.\n\n"
        cot_prompt = f" Let's think step by step. "
        few_shot_examples = self.few_shot_data[:self.num_shot] if self.num_shot else []
        template = get_template_class(self.chat_template)

        if len(few_shot_examples) == 0:
            question = [
                template.system_prompt + template.user_prompt.format(input=prompt + self.build_example_prompt(item, False)) + template.assistant_prompt.format(output="") 
                for item in data
            ]
        else:
            examples = [
                self.build_example_prompt(
                    {key: value[i] for key, value in few_shot_examples.items()}, True
                )
                for i in range(len(few_shot_examples['question']))
            ]
            question = []
            for item in data:
                request = {key: value for key, value in item.items()}
                example_prompts = examples + [self.build_example_prompt(request, False)]
                if self.cot:
                    question.append(template.system_prompt + template.user_prompt.format(input=prompt + '\n\n'.join(example_prompts)) + template.assistant_prompt.format(output=cot_prompt))
                else:
                    question.append(template.system_prompt + template.user_prompt.format(input=prompt + '\n\n'.join(example_prompts)) + template.assistant_prompt.format(output=""))
        
        return question

    
def evaluator(raw_output: List[InferenceOutput], dataloader: HybridQADataLoader, task: str):
    dataset = load_dataset(dataloader.task_dir, task)[dataloader.split]
    correct_answers = []
    responses = []
    true_cases = []
    false_cases = []
    cnt_sum = 0
    cnt_match = 0
    cnt_fail = 0

    for instance in dataset:
        correct_answers.append(
            {
                'question': instance['question'],
                'context': f"Table: {instance['table']}\nText: {instance['text']}",
                'answer': dataloader.get_answer(instance)
            }
        )
    for item in raw_output:
        responses.append(
            {
                'prompt': item.prompt,
                'response': item.response[0]
            }
        )
    for correct_answer in correct_answers:
        cnt_sum += 1
        flag_fail = True
        for response in responses:
            if correct_answer['question'] in response['prompt']:
                flag_fail = False
                eval_case = {
                    'question': correct_answer['question'],
                    'context': correct_answer['context'],
                    'correct_answer': correct_answer['answer'],
                    'response': response['response']
                }
                if judge_answer(correct_answer['answer'], response['response']):
                    cnt_match += 1
                    eval_case['result'] = True
                    true_cases.append(eval_case)
                else:
                    eval_case['result'] = False
                    false_cases.append(eval_case)
                break
        if flag_fail:
            cnt_fail += 1

    return cnt_match, cnt_sum, true_cases, false_cases

def judge_answer(correct_answer: str, response: str):
    return correct_answer.strip().lower() == response.strip().lower()

def main():
    cache_path = ".cache"
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
    
    parser = argparse.ArgumentParser(description='Evaluation Configuration')
    _, unparsed_args = parser.parse_known_args()
    keys = [k[2:] for k in unparsed_args[0::2]]
    values = list(unparsed_args[1::2])
    unparsed_args = dict(zip(keys, values))
    logger = EvalLogger('Evaluation')

    dict_configs, _ = read_eval_cfgs('hybrida', 'deepspeed')

    try:
        assert dict_configs, "Config file does not exist or is incomplete."
    except AssertionError as e:
        logger.log('error', "Config file is not exist or incomplete.")
        exit()

    for k, v in unparsed_args.items():
        if v == '' or v is None:
            continue
        dict_configs = update_dict(dict_configs, custom_cfgs_to_dict(k, v))

    dict_configs = dict_to_namedtuple(dict_configs)
    eval_configs = dict_configs.default.eval_cfgs
    logger.log_dir = eval_configs.output_dir

    dataloader = HybridQADataLoader(dict_configs)

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
        logger.print_table(title=f'HybridA/{task} Benchmark', data=eval_results)
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
