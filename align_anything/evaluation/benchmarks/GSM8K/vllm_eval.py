import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
import argparse
import json
from align_anything.evaluation.eval.base_eval import BaseEval_vllm
from align_anything.evaluation.inference.base_inference import BaseInferencer_vllm
from align_anything.evaluation.dataloader.base_dataloader import BaseDataLoader
from typing import Union, List, Dict, Any, Tuple
from datasets import load_dataset, DatasetDict
from align_anything.utils.tools import read_eval_cfgs, dict_to_namedtuple, update_dict, custom_cfgs_to_dict
from align_anything.utils.template_registry import get_template_class
from align_anything.evaluation.data_type import InferenceInput, InferenceOutput
from align_anything.evaluation.inference.base_inference import update_results
import re

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
            with open('/aifs4su/yaodong/panrui/few_shot/GSM8K/cot_few_shot/gsm8k.json', 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data
        else:
            return dataset['train']
    
    def build_example_prompt(self, data, with_answer=True, cot=False):
        answer = f'Answer: {self.get_answer(data)}' if with_answer else 'Answer: '
        return f"@@@@@@{data['question']}\n{answer}"

    def build_prompt(self, data):
        prompt = "The following are diverse grade school math word problems (with answers). Please provide the final answer after '####'.\n\n"
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
    
    def preprocess(self, data):
        prompts = self.build_prompt(data[self.split])
        
        token_ids = self.tokenizer(prompts)

        return prompts, token_ids

class GSM8KGeneratorVLLM(BaseInferencer_vllm):

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

def evaluator(raw_output: List[InferenceOutput], dataloader: GSM8KDataLoader, task: str):
    
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
                'question': instance['question'],
                'prompt_token_ids': dataloader.tokenizer(instance['question']).input_ids,
                'answer': dataloader.get_answer(instance)
            }
        )
    for item in raw_output:
        responses.append(
            {
                # 'prompt_token_ids': item.prompt_token_ids,
                'prompt_token_ids': dataloader.tokenizer(get_question_from_input(item.prompt)).input_ids,
                'generated_answer': item.response[0] if item.response else ""
            }
        )
    for correct_answer in correct_answers:
        cnt_sum += 1
        for response in responses:
            if correct_answer['prompt_token_ids'] == response['prompt_token_ids']:
                flag_fail = False
                answer = get_correct_answer(correct_answer['answer'])
                generated_answer = get_generated_answer(response['generated_answer'])
                eval_case = {
                    'question': correct_answer['question'],
                    'correct_answer': correct_answer['answer'],
                    'generated_answer': response['generated_answer']
                }
                if generated_answer == answer:
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

def get_question_from_input(input):
    prefix = '@@@@@@'
    len_prefix = len(prefix)
    index_head = input.rfind(prefix)
    index_tail = input[index_head + len_prefix:].find('\n')
    return input[index_head + len_prefix:][:index_tail]

def get_last_number(data):
    numbers = re.findall(r'\d+', data)
    if numbers:
        return numbers[-1]
    else:
        return ''
    
def get_correct_answer(data):
    index = data.rfind('####') + len('####')
    return data[index:].strip()

def get_generated_answer(data):
    if '####' in data:
        return get_last_number(get_correct_answer(data))
    else:
        return get_last_number(data)

def get_chosen_answer(logprobs: List[Dict[str, Any]], candidate_answers: List[str]):
    answer_logprobs = {}
    for logprob in logprobs:
        key = next(iter(logprob.values())).decoded_token
        value = next(iter(logprob.values())).logprob
        if key in candidate_answers:
            answer_logprobs[key] = value
    # answer_logprobs = []
    for label in candidate_answers:
        if label not in answer_logprobs.keys():
            answer_logprobs[label] = float('-inf')
    return answer_logprobs
    

def main():

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    _, unparsed_args = parser.parse_known_args()
    keys = [k[2:] for k in unparsed_args[0::2]]
    values = list(unparsed_args[1::2])
    unparsed_args = dict(zip(keys, values))
    unparsed_args = {'output_dir': '/aifs4su/yaodong/panrui/PR/align-anything/align_anything/evaluation/meta_test_output/gsm8k'}
    dict_configs, infer_configs = read_eval_cfgs('gsm8k')
    for k, v in unparsed_args.items():
        dict_configs = update_dict(dict_configs, custom_cfgs_to_dict(k, v))
        infer_configs = update_dict(infer_configs, custom_cfgs_to_dict(k, v))
    
    dict_configs, infer_configs = dict_to_namedtuple(dict_configs), dict_to_namedtuple(infer_configs)
    model_config = dict_configs.default.model_cfgs
    eval_configs = dict_configs.default.eval_cfgs
    dataloader = GSM8KDataLoader(dict_configs)
    test_data = dataloader.load_dataset()
    eval_module = GSM8KGeneratorVLLM(model_config, infer_configs)
    raw_outputs = eval_module.eval(test_data, eval_configs)

    for task, _ in raw_outputs.items():
        print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        print('task: ', task)
        print('few_shot: ', eval_configs.n_shot)
        # print('cot: ', )
        print('-----------------------------------------------------------')
        cnt_match, cnt_sum, true_cases, false_cases = evaluator(raw_outputs[task], dataloader, task)
        print('num_match: ', cnt_match, '| num_sum: ', cnt_sum, '| acc: ', cnt_match / cnt_sum)
        with open('/aifs4su/yaodong/panrui/PR/align-anything/align_anything/evaluation/benchmarks/output/GSM8K_eval.txt', 'w') as f:
            f.write(f"cnt_match: {cnt_match}\n")
            f.write(f"cnt_sum: {cnt_sum}\n")
            f.write(f"acc: {cnt_match / cnt_sum}\n")
        if true_cases:
            print('==============================TRUE CASE==============================')
            print('Question: ', true_cases[0]['question'])
            print('Correct Answer: ', true_cases[0]['correct_answer'])
            print('Generated answer: ',  true_cases[0]['generated_answer'])
        if false_cases:
            print('==============================FALSE CASE==============================')
            print('Question: ', false_cases[0]['question'])
            print('Correct Answer: ', false_cases[0]['correct_answer'])
            print('Generated answer: ',  false_cases[0]['generated_answer'])


if __name__ == '__main__':
    main()
