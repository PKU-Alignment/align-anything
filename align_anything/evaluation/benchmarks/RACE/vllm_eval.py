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

class RACEDataLoader(BaseDataLoader):

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
            with open('/aifs4su/yaodong/panrui/align-anything-evaluation/align_anything/evaluation/benchmarks/RACE/cot_few_shot/' + task + '.json', 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data
        else:
            return dataset['validation']
        
    def build_example_prompt(self, data, with_answer=True, cot=False):
        choices = '\n'.join([f'({label}) {data["options"][ord(label) - 65]}' for label in self.candidate_labels])
        answer = f'Answer: ({self.get_answer(data)})' if with_answer else 'Answer: '
        return f"{data['article']}\n\n{data['question']}\n{choices}\n{answer}"

    def build_prompt(self, data):
        prompt = f"The following is passage (with multiple choice questions and answer).\n\n"
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

class RACEGeneratorVLLM(BaseInferencer_vllm):

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

def evaluator(raw_output: List[InferenceOutput], dataloader: RACEDataLoader, task: str):
    
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
                'prompt': instance['question'],
                'prompt_token_ids': dataloader.tokenizer(instance['question']).input_ids,
                'choices': instance['options'],
                'answer': dataloader.get_answer(instance)
            }
        )
    for item in raw_output:
        responses.append(
            {
                # 'prompt_token_ids': item.prompt_token_ids,
                'prompt_token_ids': dataloader.tokenizer(get_question_from_input(item.prompt)).input_ids,
                'answer_logprobs': get_chosen_answer(item.response_logprobs[0], dataloader.candidate_labels)
            }
        )
    for correct_answer in correct_answers:
        cnt_sum += 1
        for response in responses:
            if correct_answer['prompt_token_ids'] == response['prompt_token_ids']:
                flag_fail = False
                chosen_answer = max(response['answer_logprobs'], key=response['answer_logprobs'].get)
                eval_case = {
                    'question': correct_answer['prompt'],
                    'choices': correct_answer['choices'],
                    'correct_answer': correct_answer['answer'],
                    'answer_logprobs': response['answer_logprobs'],
                    'chosen_answer': chosen_answer
                }
                if correct_answer['answer'] == chosen_answer:
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
    token_head = input.rfind('\n\n')
    index_head = input[:token_head].rfind('\n\n')
    index_tail = input[index_head + 2:].find('\n')
    return input[index_head + 2:][:index_tail]

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
    unparsed_args = {'output_dir': '/aifs4su/yaodong/panrui/align-anything-evaluation/align_anything/evaluation/meta_test_output/race'}
    dict_configs, infer_configs = read_eval_cfgs('race')
    for k, v in unparsed_args.items():
        dict_configs = update_dict(dict_configs, custom_cfgs_to_dict(k, v))
        infer_configs = update_dict(infer_configs, custom_cfgs_to_dict(k, v))
    
    dict_configs, infer_configs = dict_to_namedtuple(dict_configs), dict_to_namedtuple(infer_configs)
    model_config = dict_configs.default.model_cfgs
    eval_configs = dict_configs.default.eval_cfgs
    dataloader = RACEDataLoader(dict_configs)
    assert not (dataloader.num_shot > 0 and dataloader.cot), "Few-shot and chain-of-thought cannot be used simultaneously for this benchmark."
    test_data = dataloader.load_dataset()
    eval_module = RACEGeneratorVLLM(model_config, infer_configs)
    raw_outputs = eval_module.eval(test_data, eval_configs)

    all_cnt_match, all_cnt_sum = 0, 0
    for task, _ in raw_outputs.items():
        print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        print('task: ', task)
        print('few_shot: ', eval_configs.n_shot)
        # print('cot: ', )
        print('-----------------------------------------------------------')
        cnt_match, cnt_sum, true_cases, false_cases = evaluator(raw_outputs[task], dataloader, task)
        print('num_match: ', cnt_match, '| num_sum: ', cnt_sum, '| acc: ', cnt_match / cnt_sum)
        print('==============================TRUE CASE==============================')
        print('Question: ', true_cases[0]['question'])
        print('Choices: ', true_cases[0]['choices'])
        print('Correct Answer: ', true_cases[0]['correct_answer'])
        print('Logprobs of First Token:', true_cases[0]['answer_logprobs'])
        print('Chosen Answer',  true_cases[0]['chosen_answer'])
        print('==============================FALSE CASE==============================')
        print('Question: ', false_cases[0]['question'])
        print('Choices: ', false_cases[0]['choices'])
        print('Correct Answer: ', false_cases[0]['correct_answer'])
        print('Logprobs of First Token:', false_cases[0]['answer_logprobs'])
        print('Chosen Answer',  false_cases[0]['chosen_answer'])
        all_cnt_match += cnt_match
        all_cnt_sum += cnt_sum
    print('all_cnt_match: ', all_cnt_match, '| all_cnt_sum: ', all_cnt_sum, '| all_acc: ', all_cnt_match / all_cnt_sum)
    with open('/aifs4su/yaodong/panrui/align-anything-evaluation/align_anything/evaluation/benchmarks/output/RACE_eval.txt', 'w') as f:
        f.write(f"all_cnt_match: {all_cnt_match}\n")
        f.write(f"all_cnt_sum: {all_cnt_sum}\n")
        f.write(f"all_acc: {all_cnt_match / all_cnt_sum}\n")

if __name__ == '__main__':
    main()
