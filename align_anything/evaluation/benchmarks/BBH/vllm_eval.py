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
from align_anything.evaluation.inference.vllm_inference import *
from align_anything.evaluation.dataloader.base_dataloader import BaseDataLoader
from typing import List, Dict
from align_anything.utils.tools import read_eval_cfgs, dict_to_namedtuple, update_dict, custom_cfgs_to_dict, save_raw_outputs, load_raw_outputs
from align_anything.utils.template_registry import get_template_class
from align_anything.evaluation.data_type import InferenceInput, InferenceOutput
from datasets import load_dataset
from align_anything.evaluation.eval_logger import EvalLogger
import string
import json
import re

class BBHDataLoader(BaseDataLoader):
    def get_task_names(self):
        if isinstance(self.data_cfgs.task, list):
            return self.data_cfgs.task
        else:
            task_names = [
            self.data_cfgs.task
            ]
            return task_names

    def get_answer(self, data):
        return data['target']

    def set_fewshot_dataset(self, dataset, task=None):
        if self.cot:
            few_shot_examples = json.load(open("../cot_fewshot/BBH/" + task + ".json", encoding='utf-8'))
        else:
            few_shot_examples = []
        return few_shot_examples

    def build_example_prompt(self, data, with_answer=True):
        answer = f'Answer: {self.get_answer(data)}' if with_answer else 'Answer: '
        return f"{data['input']}\n{answer}"

    def build_prompt(self, data):
        prompt = f"The following are questions (with answers).\n\n"
        cot_prompt = f" Let's think step by step. "
        few_shot_examples = self.few_shot_data[:self.num_shot] if self.num_shot else []
        template = get_template_class(self.chat_template)
        if len(few_shot_examples) == 0:
            question = [template.system_prompt + template.user_prompt.format(input=prompt + self.build_example_prompt(item, False)) + template.assistant_prompt.format(output="") for item in data]
        else:
            few_shots = [self.build_example_prompt(example, True)for example in few_shot_examples]
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

class BBHGeneratorVLLM(BaseInferencer_vllm):
    def eval(self, data:Dict[str, List[InferenceInput]], eval_configs) -> Dict[str, List[InferenceOutput]]:
        task2details = {}
        for task, input in data.items():
            task2details[task] = self.generation(input) 
        return task2details

def is_ordered_substrings(long_str, substrings):
    last_index = 0
    for substring in substrings:
        index = long_str.find(substring)
        if index == -1 or index < last_index:
            return False
        last_index = index
    return True

def get_options(input_string):
    lines = input_string.lower().split('\n') 
    noptions_index = lines.index('options:')  
    options = lines[noptions_index+1:]  
    return options

num_to_word = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty"]
words10 = ["no matching aidnoisdsnfo", "no matching sdunoisndosds", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
def num_to_string(number):
    if(number <= 20):
        return num_to_word[number]
    if(number < 100):
        return words10[number // 10] + (" " + num_to_word[number % 10] if number % 10 != 0 else "")
    if(number < 1000):
        return num_to_word[number // 100] + " hundred" + (" and " + num_to_string(number % 100) if number % 100 != 0 else "")
    if(number < 1000000):
        return num_to_string(number // 1000) + " thousand" + (" " + num_to_string(number % 1000) if number % 1000 != 0 else "")
    else:
        return "infinity"

def get_question_type(input, target):
    input = input.lower()
    target = target.lower()
    if("tell the truth?" in input):
        return "web of lies"
    if(target == "false" or target == "true"):
        return "true/false"
    if("sort the following words alphabetically" in input):
        return "word_sorting"
    if(target == "yes" or target == "no"):
        return "yes/no"
    if("options:" in input):
        return "multiple_choice"
    if((target.isdigit() or target[1:].isdigit()) and input[-1] == "="):
        return "arithmetic"
    elif(target.isdigit()):
        return "count"
    if(all(c in '()<>[]{} ' for c in target)):
        return "parentheses"
    raise ValueError(f"Unknown question type for input: {input} and target: {target}")

def check_ans(input, target, output):
    input = input.lower()
    target = target.lower()
    output = output.lower()
    if output == target:
        return True
    type = get_question_type(input, target)
    if(type == "multiple_choice"):
        if output == target[1:2] or output.strip() == target[1:2]:
            return True
        options = get_options(input)
        opt = ""
        for _ in options:
            if target in _:
                opt = _
                break
        f = 1
        for s in options:
            if s == opt and opt not in output:
                f = 0
            if s != opt and s in output:
                f = 0
        if f == 1:
            return True
        f = 1
        for s in options:
            if s == opt and opt[4:] not in output:
                f = 0
            if s != opt and s[4:] in output:
                f = 0
        if f == 1:
            return True
        f = 1
        for s in options:
            if s == opt and opt[1:] not in output:
                f = 0
            if s != opt and s[1:] in output:
                f = 0
        if f == 1:
            return True
        f = 1
        for s in options:
            if s == opt and opt[:3] not in output:
                f = 0
            if s != opt and s[:3] in output:
                f = 0
        if f == 1:
            return True
        opt_content = opt[4:]
        if opt != "" and opt in output:
            return True
        if target == "valid":
            if "invalid" in output:
                return False
            elif "valid" in output:
                return True
        if "answer is" in output:
            ans = output.split("answer is")[1].strip()
            answers = ans.split(" ")
            if answers:
                tmp = answers[0]
                tmp = tmp.rstrip(string.punctuation)
                tmp = tmp.lstrip(string.punctuation)
                if(tmp == target):
                    return True
                if tmp == target[1:2]:
                    return True
                if tmp == opt_content:
                    return True
        if "answer:" in output:
            ans = output.split("answer:")[1].strip()
            answers = ans.split(" ")
            if answers:
                tmp = answers[0]
                tmp = tmp.rstrip(string.punctuation)
                tmp = tmp.lstrip(string.punctuation)
                if(tmp == target):
                    return True
                if tmp == target[1:2]:
                    return True
                if tmp == opt_content:
                    return True
        if "answer is:" in output:
            ans = output.split("answer is:")[1].strip()
            answers = ans.split(" ")
            if answers:
                tmp = answers[0]
                tmp = tmp.rstrip(string.punctuation)
                tmp = tmp.lstrip(string.punctuation)
                if(tmp == target):
                    return True
                if tmp == target[1:2]:
                    return True
                if tmp == opt_content:
                    return True
        return False
    if type == "true/false":
        if target not in output:
            return False
        if "true" in output and "false" not in output:
            return target == "true"
        if "false" in output and "true" not in output:
            return target == "false"
        last_true = output.rfind('true')
        last_false = output.rfind('false')
        if last_true > last_false:
            return target == "true"
        elif last_false > last_true:
            return target == "false"
    if type == "yes/no":
        if target not in output:
            return False
        if "yes" in output and "no" not in output:
            return target == "yes"
        if "no" in output and "yes" not in output:
            return target == "no"
        last_yes = output.rfind('yes')
        last_no = output.rfind('no')
        if last_yes > last_no:
            return target == "yes"
        elif last_no > last_yes:
            return target == "no"
    if type == "web of lies":
        pattern = r'does \w+ tell the truth'
        match = re.findall(pattern, input)
        
        last_sentence = match[-1]  

        name = last_sentence.split(' ')[1]
        
        form1 = f'{name} tells the truth'
        form2 = f'{name} is telling the truth'
        form3 = f'{name} doesn\'t tell the truth'
        form4 = f'{name} does not tell the truth'
        form5 = f'{name} is lying'
        form6 = f'{name} is a liar'
        form7 = f'{name} is not telling the truth'
        form8 = f'{name} does tell the truth'
        if form1 in output or form2 in output or form8 in output:
            return target == "yes"
        if form3 in output or form4 in output or form5 in output or form6 in output or form7 in output:
            return target == "no"
        if re.search(r'\byes\b', output) and not re.search(r'\bno\b', output):
            return target == "yes"
        if re.search(r'\bno\b', output) and not re.search(r'\byes\b', output):
            return target == "no"
        if "yes" in output and "no" in output:
            return output.rindex("yes") < output.rindex("no")
        return False
    if type == "word_sorting":
        return is_ordered_substrings(output, target.split())
    if type == "count":
        words = output.split()
        for i in range(len(words) - 2):
            if words[i] == "answer" and words[i+1] == "is" and words[i+2].isdigit():
                return int(words[i+2]) == int(target)
        numbers = re.findall(r'\d+', output)
        if not numbers:
            num_string = num_to_string(int(target))
            if num_string in output:
                substrings = output.split(num_string)
                if all(words not in substrings[-1] for words in num_to_word) and all(words not in substrings[1] for words in words10):
                    return True
            return False
        return max(map(int, numbers)) == int(target)
    if type == "arithmetic":
        words = output.split()
        for i in range(len(words) - 2):
            if words[i] == "answer" and words[i+1] == "is" and words[i+2].isdigit():
                return int(words[i+2]) == int(target)
        numbers = re.findall(r'-?\d+', output)
        if not numbers:
            return False
        return int(numbers[-1]) == int(target)
    if type == "parentheses":
        s = output.replace(" ", "")
        t = target.replace(" ", "")
        i = input.replace(" ", "") 
        substrings = re.findall(r'[\(\)\[\]\{\}\<\>]+', s)
        ori_string = re.findall(r'[\(\)\[\]\{\}\<\>]+', i)[-1]
        full = ori_string + t
        if len(substrings) == 0:
            return False
        return t in substrings or full in substrings
    return False

def evaluator(raw_output: List[InferenceOutput], dataloader: BBHDataLoader, task: str, file_path):
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
                'prompt': instance['input'],
                'answer': dataloader.get_answer(instance)
            }
        )
    for output in raw_output:
        responses.append(
            {
                'prompt': (output.prompt),
                'answer': output.response[0]
            }
        )
    for correct_answer in correct_answers:
        cnt_sum += 1
        for response in responses:
            if correct_answer['prompt'] in response['prompt']:
                flag_fail = False
                true_or_false = check_ans(correct_answer['prompt'], correct_answer['answer'], response['answer'])
                if true_or_false:
                    cnt_match += 1
                save_detail(correct_answer['prompt'], '', correct_answer['answer'], response['answer'], true_or_false, file_path)
                break
        if flag_fail:
            cnt_fail += 1
        else:
            flag_fail = True

    return cnt_match , cnt_sum

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    _, unparsed_args = parser.parse_known_args()
    keys = [k[2:] for k in unparsed_args[0::2]]
    values = list(unparsed_args[1::2])
    unparsed_args = dict(zip(keys, values))
    dict_configs, infer_configs = read_eval_cfgs('bbh', 'vLLM')

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
    dataloader = BBHDataLoader(dict_configs)
    assert not (dataloader.num_shot > 0 and not dataloader.cot), "Few-shot cannot be used without chain-of-thought for this benchmark."
    test_data = dataloader.load_dataset()
    eval_module = BBHGeneratorVLLM(model_config, infer_configs)
    raw_outputs_dir = os.path.join(eval_configs.output_dir, f"raw_outputs_{re.sub(r'/', '_', model_config.model_name_or_path)}.pkl")
    if os.path.exists(raw_outputs_dir):
        raw_outputs = load_raw_outputs(raw_outputs_dir)
    else:
        raw_outputs = eval_module.eval(test_data, eval_configs)
        save_raw_outputs(raw_outputs, raw_outputs_dir)
  
    os.makedirs(logger.log_dir, exist_ok=True)
    uuid_path = f"{logger.log_dir}/{eval_configs.uuid}"
    os.makedirs(uuid_path, exist_ok=True)

    tot_num_match, tot_num_sum = 0, 0
    for task, _ in raw_outputs.items():

        file_path = f"{uuid_path}/{task}.json"
        cnt_match, cnt_sum = evaluator(raw_outputs[task], dataloader, task, file_path)
        tot_num_match += cnt_match
        tot_num_sum += cnt_sum

        eval_results = {
            'model_id': [dict_configs.default.model_cfgs.model_id],
            'num_fewshot': [eval_configs.n_shot],
            'chain_of_thought': [eval_configs.cot],
            'num_match': [cnt_match],
            'num_sum': [cnt_sum],
            'accuracy': [cnt_match / cnt_sum]
        }
        logger.print_table(title=f'BBH/{task} Benchmark', data=eval_results)
        logger.log('info', '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        logger.log('info', f"task: {task}")
        logger.log('info', f"model_id: {eval_results['model_id'][0]},")
        logger.log('info', f"num_fewshot: {eval_results['num_fewshot'][0]},")
        logger.log('info', f"chain_of_thought: {eval_results['chain_of_thought'][0]},")
        logger.log('info', f"num_match: {eval_results['num_match'][0]},")
        logger.log('info', f"num_sum: {eval_results['num_sum'][0]},")
        logger.log('info', f"accuracy: {eval_results['accuracy'][0]},")
        logger.log('info', '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

    eval_results = {
        'model_id': [dict_configs.default.model_cfgs.model_id],
        'num_fewshot': [eval_configs.n_shot],
        'chain_of_thought': [eval_configs.cot],
        'tot_num_match': [tot_num_match],
        'tot_num_sum': [tot_num_sum],
        'tot_accuracy': [tot_num_match / tot_num_sum]
    }
    logger.print_table(title=f'BBH Benchmark', data=eval_results)
    logger.log('info', '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    logger.log('info', f"model_id: {eval_results['model_id'][0]},")
    logger.log('info', f"num_fewshot: {eval_results['num_fewshot'][0]},")
    logger.log('info', f"chain_of_thought: {eval_results['chain_of_thought'][0]},")
    logger.log('info', f"tot_num_match: {eval_results['tot_num_match'][0]},")
    logger.log('info', f"tot_num_sum: {eval_results['tot_num_sum'][0]},")
    logger.log('info', f"tot_accuracy: {eval_results['tot_accuracy'][0]},")
    logger.log('info', '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

if __name__ == '__main__':
    main()
