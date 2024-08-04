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

import csv
import os

from datasets import DatasetDict, Dataset
from tqdm import tqdm

import evaluator
import utils2
from utils2 import get_time, BaseEval, random_seed
import numpy as np
import pandas as pd
import torch
import json
from collections import defaultdict
import re
import glob
import random
import os.path as osp
import time
import datasets
import collections
import traceback
import ppl_metric_eval
# import legalbench as zxrlb
# from legalbench import tasks as zxrlb_tasks
# from legalbench import utils as zxrlb_utils
# from legalbench import evaluation as zxrlb_evaluation
from collections import OrderedDict
import math
import copy



random.seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.manual_seed(random_seed)


class MMLUTrans(BaseEval):
    def get_task_names(self):
        subjects = list(utils2.MMLUSubcategories.keys())
        return subjects
    
    def get_answer(self, data):
        return data['answer'] if 'answer' in data else 'N'   

    def set_fewshot_dataset(self, dataset):
        self.few_shot_data = dataset['dev']
    
    def loaddataset(self, task_name, *args):
        def pf(data):
            new_data = []
            i = 0
            for x in data:
                try:
                    y = json.loads(x)
                    new_data.append(y)
                except:
                    i += 1
                    pass
            if i > 0:
                print(f'Failed num {i}')
            return new_data

        print(task_name)
        test = json.load(open(os.path.join(self.data_path, 'test.json'), encoding='utf-8'))[task_name]
        dev = json.load(open(os.path.join(self.data_path, 'dev.json'), encoding='utf-8'))[task_name]
        test = pf(test)
        dev = pf(dev)
        mydataset = datasets.DatasetDict({
            'test': datasets.Dataset.from_list(test),
            'dev': datasets.Dataset.from_list(dev),
        })
        return mydataset

    def build_an_example_prompt(self, data, with_answer=True):
        if 'question' in data:
            question = data['question']
        elif '问题' in data:
            question = data['问题']
        else:
            raise
        choice = [f'{label}: {data[label]}' for label in self.candidate_labels]
        cstr = '\n'.join(choice)
        answer = self.get_answer(data) if with_answer else ''
        return f"{question}\n{cstr}\nAnswer:{answer}"
        
    def build_prompt(self, task_name, data, dataset, shot):
        prompt = f"The following are multiple choice questions (with answers) about {task_name.replace('_', ' ')}.\n\n"
        if shot == 0:
            prompt += f"{self.build_an_example_prompt(data, with_answer=False)}"
        else:
            few_shot_data = self.few_shot_data
            for i in range(min(len(few_shot_data), shot)):
                prompt += self.build_an_example_prompt(few_shot_data[i], with_answer=True) + '\n\n'
            prompt += self.build_an_example_prompt(data, with_answer=False)
        return prompt


class CMMLUTrans(BaseEval):
    def get_task_names(self):
        subjects = list(utils2.CMMLUTasks.keys())
        return subjects
    
    def get_answer(self, data):
        return data['answer'] if 'answer' in data else 'N'    
    
    def set_fewshot_dataset(self, dataset):
        self.few_shot_data = dataset['dev']
    
    def loaddataset(self, task_name, *args):
        def pf(data):
            new_data = []
            i = 0
            for x in data:
                try:
                    y = json.loads(x)
                    new_data.append(y)
                except:
                    i += 1
                    pass
            if i > 0:
                print(f'Failed num {i}')
            return new_data

        print(task_name)
        test = json.load(open(os.path.join(self.data_path, 'test.json'), encoding='utf-8'))[task_name]
        dev = json.load(open(os.path.join(self.data_path, 'dev.json'), encoding='utf-8'))[task_name]
        test = pf(test)
        dev = pf(dev)
        mydataset = datasets.DatasetDict({
            'test': datasets.Dataset.from_list(test),
            'dev': datasets.Dataset.from_list(dev),
        })
        return mydataset

    def build_an_example_prompt(self, data, with_answer=True):
        if 'question' in data:
            question = data['question']
        elif '问题' in data:
            question = data['问题']
        else:
            raise
        choice = [f'{label}. {data[label]}' for label in self.candidate_labels]
        cstr = '\n'.join(choice)
        answer = self.get_answer(data) if with_answer else ''
        return f"Question: {question}\n{cstr}\nAnswer: {answer}"
        
    def build_prompt(self, task_name, data, dataset, shot):
        prompt = f"The following are multiple choice questions (with answers) about {task_name.replace('_', ' ')}.\n\n"
        if shot == 0:
            prompt += f"{self.build_an_example_prompt(data, with_answer=False)}"
        else:
            few_shot_data = self.few_shot_data
            for i in range(min(len(few_shot_data), shot)):
                prompt += self.build_an_example_prompt(few_shot_data[i], with_answer=True) + '\n\n'
            prompt += self.build_an_example_prompt(data, with_answer=False)
        return prompt


class MedQATrans(BaseEval):
    def get_task_names(self):
        return ['usmle', 'mcmle']
    
    def get_answer(self, data):
        return data['answer'] if 'answer' in data else 'N'    

    def set_fewshot_dataset(self, dataset):
        self.few_shot_data = dataset['dev'] 
    
    def loaddataset(self, task_name, *args):
        def pf(data):
            new_data = []
            i = 0
            for x in data:
                try:
                    y = json.loads(x)
                    new_data.append(y)
                except:
                    i += 1
                    pass
            if i > 0:
                print(f'Failed num {i}')
            return new_data

        print(task_name)
        test = json.load(open(os.path.join(self.data_path, 'test.json'), encoding='utf-8'))[task_name]
        dev = json.load(open(os.path.join(self.data_path, 'validation.json'), encoding='utf-8'))[task_name]
        test = pf(test)
        dev = pf(dev)
        mydataset = datasets.DatasetDict({
            'test': datasets.Dataset.from_list(test),
            'dev': datasets.Dataset.from_list(dev),
        })
        return mydataset

    def build_an_example_prompt(self, data, with_answer=True):
        if 'question' in data:
            question = data['question']
        elif '问题' in data:
            question = data['问题']
        else:
            raise
        choice = [f'{label}. {data[label]}' for label in self.candidate_labels]
        cstr = '\n'.join(choice)
        answer = self.get_answer(data) if with_answer else ''
        return f"{question}\n{cstr}\nAnswer: {answer}"
        
    def build_prompt(self, task_name, data, dataset, shot):
        if shot == 0:
            prompt = f"{self.build_an_example_prompt(data, with_answer=False)}"
        else:
            few_shot_data = self.few_shot_data
            prompt = ""
            for i in range(min(len(few_shot_data), shot)):
                prompt += '\n' + self.build_an_example_prompt(few_shot_data[i], with_answer=True)
            prompt += '\n' + self.build_an_example_prompt(data, with_answer=False)
        return prompt


class CRUXEval(BaseEval):
    def get_task_names(self):
        self.cot = False
        return []
    
    def set_fewshot_dataset(self, dataset):
        self.few_shot_data = None

    def make_direct_output_prompt_phind(self, s):
        code, input = s
        return f"""Based on the given Python code, which may contain errors, complete the assert statement with the output when executing the code on the given test case. Do NOT output any extra information, even if the function is incorrect or incomplete. Output "# done" after the assertion.

def f(n):
    return n
assert f(17) == 17 # done

def f(s):
    return s + "a"
assert f("x9j") == "x9ja" # done

{code}
assert f({input}) =="""

    def make_cot_output_prompt(self, s):
        code, input = s
        return f"""You are given a Python function and an assertion containing an input to the function. Complete the assertion with a literal (no unsimplified expressions, no function calls) containing the output when executing the provided code on the given input, even if the function is incorrect or incomplete. Do NOT output any extra information. Execute the program step by step before arriving at an answer, and provide the full assertion with the correct output in [ANSWER] and [/ANSWER] tags, following the examples.

[PYTHON]
def f(s):
    s = s + s
    return "b" + s + "a"
assert f("hi") == ??
[/PYTHON]
[THOUGHT]
Let's execute the code step by step:

1. The function f is defined, which takes a single argument s.
2. The function is called with the argument "hi", so within the function, s is initially "hi".
3. Inside the function, s is concatenated with itself, so s becomes "hihi".
4. The function then returns a new string that starts with "b", followed by the value of s (which is now "hihi"), and ends with "a".
5. The return value of the function is therefore "bhihia".
[/THOUGHT]
[ANSWER]
assert f("hi") == "bhihia"
[/ANSWER]

[PYTHON]
{code}
assert f({input}) == ??
[/PYTHON]
[THOUGHT]
"""

    def make_direct_output_prompt(self, s):
        code, input = s
        return f"""You are given a Python function and an assertion containing an input to the function. Complete the assertion with a literal (no unsimplified expressions, no function calls) containing the output when executing the provided code on the given input, even if the function is incorrect or incomplete. Do NOT output any extra information. Provide the full assertion with the correct output in [ANSWER] and [/ANSWER] tags, following the examples.

[PYTHON]
def f(n):
    return n
assert f(17) == ??
[/PYTHON]
[ANSWER]
assert f(17) == 17
[/ANSWER]

[PYTHON]
def f(s):
    return s + "a"
assert f("x9j") == ??
[/PYTHON]
[ANSWER]
assert f("x9j") == "x9ja"
[/ANSWER]

[PYTHON]
{code}
assert f({input}) == ??
[/PYTHON]
[ANSWER]
"""

    def make_direct_input_prompt(self, s):
        code, output = s
        return f"""You will be given a function f and an output in the form f(??) == output. Find any input such that executing f on the input leads to the given output. There may be multiple answers, but you should only output one. In [ANSWER] and [/ANSWER] tags, complete the assertion with one such input that will produce the output when executing the function.

[PYTHON]
def f(my_list):
    count = 0
    for i in my_list:
        if len(i) % 2 == 0:
            count += 1
    return count
assert f(??) == 3
[/PYTHON]
[ANSWER]
assert f(["mq", "px", "zy"]) == 3
[/ANSWER]

[PYTHON]
def f(s1, s2):
    return s1 + s2
assert f(??) == "banana"
[/PYTHON]
[ANSWER]
assert f("ba", "nana") == "banana"
[/ANSWER]

[PYTHON]
{code}
assert f(??) == {output}
[/PYTHON]
[ANSWER]
"""

    def make_cot_input_prompt(self, s):
        code, output = s
        return f"""You will be given a function f and an output in the form f(??) == output. Your task is to find any input such that executing f on the input leads to the given output. There may be multiple answers, but only output one. First, think step by step. You MUST surround the answer with [ANSWER] and [/ANSWER] tags. Express your answer as a passing assertion containing the input and the given output.

[PYTHON]
def f(x):
    return x + 1
assert f(??) == 17
[/PYTHON]
[THOUGHT]
To find an input such that executing f on the input leads to the given output, we can work backwards from the given assertion. We know that f(??) == 17. 

Since the function f(x) returns x + 1, for f(??) to be equal to 17, the value of ?? should be 16. 
[/THOUGHT]
[ANSWER]
assert f(16) == 17
[/ANSWER]

[PYTHON]
{code}
assert f(??) == {output}
[/PYTHON]
[THOUGHT]
"""

    def build_prompt(self, task_name, data, dataset, shot):
        pass

    def loaddataset(self, task_name, *args):
        file_path = os.path.join(self.data_path, "test.jsonl")
        # file_path = os.path.join(self.data_path, "sample_10.jsonl")
        mydataset = datasets.load_dataset('json', data_files={'test': file_path})
        return mydataset

    def is_correct(self, prediction, test_case):
        pass
        
    def parser_generation(self, continuation):
        return continuation

    def calc_acc(self):
        all_record = {
            'start_time': self.start_time,
            'end_time': get_time(),
            'md5_info': self.md5_info
        }
        from cruxeval_evaluation.evaluate_generations import evaluate_generations
        generations = json.load(open(os.path.join(self.output_dir, self.predict_filename), encoding='utf-8'))[self.task_names[0]]
        mode = self.task_names[0]
        datafilename = os.path.join(self.data_path, "test.jsonl")
        # datafilename = os.path.join(self.data_path, "sample_10.jsonl")
        results = evaluate_generations(generations, mode, datafilename)
        pass_at_1 = str(round(results["pass_at_1"], 2))
        pass_at_5 = str(round(results["pass_at_5"], 2))
        record = dict()
        record['instance_num'] = len(generations)
        record['pass_at_1'] = pass_at_1
        record['pass_at_5'] = pass_at_5
        all_record['scores'] = record
        self.save_results({self.filename_pfx: all_record})


class CRUXEvalInput(CRUXEval):
    def get_task_names(self):
        self.cot = False
        return ['input']
    
    def get_answer(self, data):
        return data['input']

    def build_prompt(self, task_name, data, dataset, shot):
        return self.make_direct_input_prompt((data['code'], data['output']))
    
    def parser_generation(self, continuation):
        results = []
        for generation in continuation:
            if self.cot:
                if "[ANSWER]" in generation:
                    generation = generation.split("[ANSWER]")[1].strip()
            if "==" in generation:
                generation = generation.split("==")[0].strip()
            if "assert" in generation:
                generation = generation.split("assert")[1].strip()
            results.append(generation.strip())
        return results


class CRUXEvalOutput(CRUXEval):
    def get_task_names(self):
        self.cot = False
        return ['output']
    
    def get_answer(self, data):
        return data['output']

    def build_prompt(self, task_name, data, dataset, shot):
        return self.make_direct_output_prompt((data['code'], data['input']))

    def parser_generation(self, continuation):
        results = []
        for generation in continuation:
            if self.cot:
                if "[ANSWER]" in generation:
                    generation = generation.split("[ANSWER]")[1].strip()
            generation = generation.split('[/ANSWER]')[0]
            if "==" in generation:
                generation = generation.split("==")[1].strip()
            results.append(generation.strip())
        return results

    

class RoleEval(BaseEval):
    def get_task_names(self):
        subnames = ['chinese', 'global']
        names = ['anime_and_comics', 'fiction', 'movies_and_tv_series', 'games', 'celebrities']
        task_names = []
        for sub in subnames:
            for n in names:
                task_names.append(f'{sub}-{n}')
        return task_names
    
    def loaddataset(self, task_name, *args):
        subname, task_name = task_name.split('-')
        df1 = pd.read_csv(os.path.join(self.data_path, subname, 'test', task_name + '_test.csv'))
        df2 = pd.read_csv(os.path.join(self.data_path, subname, "dev", task_name + '_dev.csv'))
        mydataset = datasets.DatasetDict({
            'test': datasets.Dataset.from_pandas(df1),
            'dev': datasets.Dataset.from_pandas(df2),
        })

        return mydataset
    
    def get_answer(self, data):
        return data['answer'] if 'answer' in data else 'NoAnswer'   
    
    def set_fewshot_dataset(self, dataset):
        self.few_shot_data = dataset['dev']

    def build_an_example_prompt(self, data, with_answer=True, flag=0):
        question = data['question']
        choice = [f'{a}：{data[a]}' for a in self.candidate_labels]
        cstr = '\n'.join(choice)
        answer = self.get_answer(data) if with_answer else ''
        return f"{question}\n{cstr}\n答案：{answer}"
        
    def build_prompt(self, task_name, data, dataset, shot):
        prompt = ''
        if shot == 0:
            prompt += self.build_an_example_prompt(data, with_answer=False)
        else:
            few_shot_data = self.few_shot_data
            for i in range(min(len(few_shot_data), shot)):
                prompt += self.build_an_example_prompt(few_shot_data[i], with_answer=True) + '\n'
            prompt += self.build_an_example_prompt(data, with_answer=False)
        return prompt


class Lambada(BaseEval):
    def get_task_names(self):
        task_names = ['lambada', ]
        return task_names
    
    def loaddataset(self, task_name, *args):
        print(task_name)
        filename = os.path.join(self.data_path, 'test.jsonl')
        with open(filename, encoding='utf-8') as f:
            data = [json.loads(x) for x in f.readlines()]
        dataset = datasets.DatasetDict({
            'test': datasets.Dataset.from_list(data),
        })
        return dataset
    
    def get_answer(self, data):
        return data['label'] if 'label' in data else 'NoAnswer'   
    
    def set_fewshot_dataset(self, dataset):
        self.few_shot_data = None

    def build_an_example_prompt(self, data, with_answer=True, flag=0):
        pass
        
    def build_prompt(self, task_name, data, dataset, shot):
        prompt = f"Please complete the following sentence:\n{data['prompt']}"
        return prompt

    def parser_generation(self, continuation):
        continuation = continuation[0]
        for seg in continuation.split(' '):
            seg = seg.strip()
            if len(seg) > 0:
                return seg
        return continuation
    
    def is_correct(self, pred, answer):
        return answer in pred


class Winogrande(BaseEval):
    def get_task_names(self):
        task_names = ['winogrande', ]
        return task_names
    
    def loaddataset(self, task_name, *args):
        print(task_name)
        filename = os.path.join(self.data_path, 'dev.jsonl')
        with open(filename, encoding='utf-8') as f:
            data = [json.loads(x) for x in f.readlines()]
        for d in data:
            d['id'] = d['qID']
        dataset = datasets.DatasetDict({
            'test': datasets.Dataset.from_list(data),
        })
        return dataset
    
    def get_answer(self, data):
        return 'ABCDEFG'[int(data['answer'])-1] if 'answer' in data else 'NoAnswer'   
    
    def set_fewshot_dataset(self, dataset):
        self.few_shot_data = None

    def build_an_example_prompt(self, data, with_answer=True, flag=0):
        pass
        
    def build_prompt(self, task_name, data, dataset, shot):
        assert shot == 0
        question = data['sentence']
        choices = [data[key] for key in ('option1', 'option2')]
        prompt = [question.replace('_', can) for can in choices]
        return prompt

    def parser_generation(self, continuation):
        continuation = continuation[0]
        continuation = continuation.split('\n')[0]
        return continuation


class Hellaswag(BaseEval):
    def get_task_names(self):
        task_names = ['hellaswag', ]
        return task_names
    
    def loaddataset(self, task_name, *args):
        print(task_name)
        filename = os.path.join(self.data_path, 'hellaswag.jsonl')
        with open(filename, encoding='utf-8') as f:
            data = [json.loads(x) for x in f.readlines()]
        dataset = datasets.DatasetDict({
            'test': datasets.Dataset.from_list(data)
        })
        return dataset
    
    def get_answer(self, data):
        return 'ABCDEFG'[int(data['gold'])] if 'gold' in data else 'NoAnswer'   
    
    def set_fewshot_dataset(self, dataset):
        self.few_shot_data = None

    def build_an_example_prompt(self, data, with_answer=True, flag=0):
        pass
        
    def build_prompt(self, task_name, data, dataset, shot):
        assert shot == 0
        question = data['query']
        choices = data['choices']
        prompt = [question + ' ' + can for can in choices]
        return prompt

    def parser_generation(self, continuation):
        continuation = continuation[0]
        continuation = continuation.split('\n')[0]
        return continuation


class NQ(BaseEval):
    def get_task_names(self):
        return ["nq"]

    def loaddataset(self, task_name, *args):
        print(task_name)
        dataset = DatasetDict()
        for split in ['dev', 'test']:
            filename = osp.join(self.data_path, f'nq-{split}.qa.csv')
            with open(filename, 'r', encoding='utf-8') as f:
                reader = csv.reader(f, delimiter='\t')
                raw_data = []
                for row in reader:
                    assert len(row) == 2
                    question = row[0]
                    answers = eval(row[1])
                    if split == 'dev':
                        answers = answers[0]
                    raw_data.append({'question': question, 'answer': answers})
                dataset[split] = Dataset.from_list(raw_data)
        return dataset

    def set_fewshot_dataset(self, dataset):
        self.few_shot_data = None

    def build_an_example_prompt(self, data, with_answer=True, flag=0):
        pass

    def build_prompt(self, task_name, data, dataset, shot):
        question = data["question"]
        prompt = [f'Question: {question}?\nAnswer: ']
        return prompt

    def get_answer(self, data):
        return data['answer']

    def is_correct(self, prediction, reference):
        prediction = prediction.strip().split('\n')[0].lower()
        if 'answer is' in prediction:
            prediction = prediction.split('answer is')[-1]
        processed_prediction = general_postprocess(prediction)
        processed_reference = [general_postprocess(j).lower() for j in reference]

        # is_correct = any([cand == pred for cand in cand_ans])
        is_correct = any([cand in processed_prediction for cand in processed_reference])
        return is_correct


class PIQA(BaseEval):
    def get_task_names(self):
        return ["piqa"]

    def load_single(self, path, data_filename, label_filename):
        data_path = os.path.join(path, data_filename)
        label_path = os.path.join(path, label_filename)
        dataset = []
        with open(data_path, 'r', encoding='utf-8') as f:
            data_lines = f.readlines()
        with open(label_path, 'r', encoding='utf-8') as f:
            label_lines = f.readlines()
        assert len(data_lines) == len(label_lines)
        for data, label in zip(data_lines, label_lines):
            i = json.loads(data.strip())
            i['label'] = chr(int(label.strip())+65)
            dataset.append(i)

        return Dataset.from_list(dataset)

    def loaddataset(self, task_name, *args):
        print(task_name)
        train_dataset = self.load_single(self.data_path, 'train.jsonl', 'train-labels.lst')
        dev_dataset = self.load_single(self.data_path, 'dev.jsonl', 'dev-labels.lst')
        return DatasetDict({'train': train_dataset, 'dev': dev_dataset})

    def set_fewshot_dataset(self, dataset):
        self.few_shot_data = None

    def build_an_example_prompt(self, data, with_answer=True, flag=0):
        pass

    def build_prompt(self, task_name, data, dataset, shot):
        goal = data["goal"]
        sol1 = data["sol1"]
        sol2 = data["sol2"]
        prompt = [f'The following makes sense: \nQ: {goal}\nA: {sol1}\n', f'The following makes sense: \nQ: {goal}\nA: {sol2}\n']
        return prompt

    def get_answer(self, data):
        return data['label']


class RACE(BaseEval):
    def get_task_names(self):
        # return ["race-high"]
        return ["race-middle", "race-high"]

    def loaddataset(self, task_name, *args):
        print(task_name)
        if task_name == 'race-middle':
            devfilename = os.path.join(self.data_path, 'validation/middle.jsonl')
            testfilename = os.path.join(self.data_path, 'test/middle.jsonl')
        elif task_name == 'race-high':
            devfilename = os.path.join(self.data_path, 'validation/high.jsonl')
            testfilename = os.path.join(self.data_path, 'test/high.jsonl')
        with open(devfilename, encoding='utf-8') as f:
            dev = [json.loads(x) for x in f.readlines()]
        with open(testfilename, encoding='utf-8') as f:
            test = [json.loads(x) for x in f.readlines()]
        dataset = datasets.DatasetDict({
            'test': datasets.Dataset.from_list(test),
            'dev': datasets.Dataset.from_list(dev)
        })
        return dataset

    def set_fewshot_dataset(self, dataset):
        self.few_shot_data = None

    def build_an_example_prompt(self, data, with_answer=True, flag=0):
        pass

    def build_prompt(self, task_name, data, dataset, shot):
        article = data["article"]
        question = data["question"]
        A = data["options"][0]
        B = data["options"][1]
        C = data["options"][2]
        D = data["options"][3]
        # prompt = [f"Read the article, and answer the question by replying A, B, C or D.\n\nArticle:\n{article}\n\n" \
        #           f"Q: {question}\n\nA. {A}\nB. {B}\nC. {C}\nD. {D}\n\nA: {ans}" for ans in ['A', 'B', 'C', 'D']]
        prompt = [f"Read the article, and answer the question by replying A, B, C or D.\n\nArticle:\n{article}\n\n" \
                  f"Question: {question}\n\nA. {A}\nB. {B}\nC. {C}\nD. {D}\n\nAnswer: {ans}" for ans in ['A', 'B', 'C', 'D']]
        return prompt

    def get_answer(self, data):
        return data['answer']


class ARC(BaseEval):
    def get_task_names(self):
        task_names = ['ARC-c', 'ARC-e']
        return task_names
    
    def loaddataset(self, task_name, *args):
        print(task_name)
        if task_name == 'ARC-c':
            devfilename = os.path.join(self.data_path, f'{task_name}', 'ARC-Challenge-Dev.jsonl')
            testfilename = os.path.join(self.data_path, f'{task_name}', 'ARC-Challenge-Test.jsonl')
        if task_name == 'ARC-e':
            devfilename = os.path.join(self.data_path, f'{task_name}', 'ARC-Easy-Dev.jsonl')
            testfilename = os.path.join(self.data_path, f'{task_name}', 'ARC-Easy-Test.jsonl')
        with open(devfilename, encoding='utf-8') as f:
            dev = [json.loads(x) for x in f.readlines()]
        with open(testfilename, encoding='utf-8') as f:
            test = [json.loads(x) for x in f.readlines()]
        dataset = datasets.DatasetDict({
            'test': datasets.Dataset.from_list(test),
            'dev': datasets.Dataset.from_list(dev)
        })
        return dataset
    
    def get_answer(self, data):
        return data['answerKey'] if 'answerKey' in data else 'NoAnswer'   
    
    def set_fewshot_dataset(self, dataset):
        self.few_shot_data = None

    def build_an_example_prompt(self, data, with_answer=True, flag=0):
        question = data['question']['stem']
        choices = data['question']['choices']
        choices = list(sorted(choices, key=lambda x: x['label']))
        cstr = '\n'.join([f"{can['label']}: {can['text']}" for can in choices])
        answer = self.get_answer(data) if with_answer else ''
        return f"{question}\n{cstr}\nAnswer: {answer}"
        
    def build_prompt(self, task_name, data, dataset, shot):
        assert shot == 0
        question = data['question']['stem']
        choices = data['question']['choices']
        choices = list(sorted(choices, key=lambda x: x['label']))
        prompt = [f"Question: {question}\nAnswer: {can['text']}" for can in choices]
        return prompt

    def parser_generation(self, continuation):
        continuation = continuation[0]
        continuation = continuation.split('\n')[0]
        return continuation

class BBH(BaseEval):
    def get_task_names(self):
        task_names = [] + utils2.BBH_multiple_choice_sets + utils2.BBH_free_form_sets
        return task_names
    
    def loaddataset(self, task_name, *args):
        print(task_name)
        filename = os.path.join(self.data_path, 'data', f'{task_name}.json')
        data = json.load(open(filename, encoding='utf-8'))['examples']
        dataset = datasets.DatasetDict({
            'test': datasets.Dataset.from_list(data[:])
        }) 
    
        with open(os.path.join(self.data_path, 'lib_prompt', f'{task_name}.txt')) as f:
            self.hints = ''.join(f.readlines())
        return dataset
    
    def get_answer(self, data):
        return data['target'] if 'target' in data else 'NoAnswer'   
    
    def set_fewshot_dataset(self, dataset):
        self.few_shot_data = None 
        
    def build_prompt(self, task_name, data, dataset, shot):
        prompt = 'Follow the given examples and answer the question.'
        prompt += f"{prompt}\n{self.hints}\n\nQ: {data['input']}\nA: Let's think step by step."
        return prompt

    def parser_generation(self, continuation):
        continuation = continuation[0]
        continuation = continuation.split('\n\n')[0]
        pred = continuation.replace('.', '').split('So the answer is ')[-1]
        return pred


class XingCe(BaseEval):
    def get_task_names(self):
        task_names = ['judge_and_reasoning', 'quantitative_relationship']
        return task_names
    
    def loaddataset(self, task_name, *args):
        filename = os.path.join(self.data_path, 'data.json')
        datalist = json.load(open(filename, encoding='utf-8'))
        if task_name == 'judge_and_reasoning':
            datalist = list(filter(lambda x: x['category'] == '判断推理', datalist))
        elif task_name == 'quantitative_relationship':
            datalist = list(filter(lambda x: x['category'] == '数量关系', datalist))
        val = datalist[: 5]
        test = datalist[5: ]
        dataset = datasets.DatasetDict({
            'test': datasets.Dataset.from_list(test),
            'val': datasets.Dataset.from_list(val)
        })
        return dataset
    
    def get_answer(self, data):
        return data['answer'] if 'answer' in data else 'NoAnswer'   
    
    def set_fewshot_dataset(self, dataset):
        self.few_shot_data = dataset['val'].shuffle(seed=random_seed) 

    def build_an_example_prompt(self, data, with_answer=True, flag=0):
        question = data['question']
        answer = data['answer'] if with_answer else ''
        return f"{question}\n答案：{answer}"
        
    def build_prompt(self, task_name, data, dataset, shot):
        prompt = ''
        if shot == 0:
            prompt += self.build_an_example_prompt(data, with_answer=False)
        else:
            few_shot_data = self.few_shot_data
            for i in range(min(len(few_shot_data), shot)):
                prompt += self.build_an_example_prompt(few_shot_data[i], with_answer=True) + '\n'
            prompt += self.build_an_example_prompt(data, with_answer=False)
        return prompt
    
class XingCeTL(XingCe):
    def get_task_names(self):
        task_names = ['judge_and_reasoning', ]
        return task_names


class XingCeSL(XingCe):
    def get_task_names(self):
        task_names = ['quantitative_relationship', ]
        return task_names


class ItemCount(BaseEval):
    def get_task_names(self):
        task_names = ['itemcount']
        return task_names
    
    def loaddataset(self, task_name, *args):
        val = json.load(open(
            os.path.join(self.data_path, 'val.json'), encoding='utf-8'))
        test = json.load(open(
            os.path.join(self.data_path, 'test.json'), encoding='utf-8'))        
        dataset = datasets.DatasetDict()
        dataset['val'] = datasets.Dataset.from_list(val)
        dataset['test'] = datasets.Dataset.from_list(test)

        return dataset
    
    def get_answer(self, data):
        return self.candidate_labels[data['candidates'].index(data['answer'])] if 'answer' in data else 'NoAnswer'   
    
    def set_fewshot_dataset(self, dataset):
        self.few_shot_data = dataset['val'].shuffle(seed=random_seed) 

    def build_an_example_prompt(self, data, with_answer=True, flag=0):
        context = data['context']
        question = data['question']
        choice = [f'{a}：{b}' for a, b in zip(self.candidate_labels, data['candidates'])]
        cstr = '\n'.join(choice)
        answer = data['answer'] if with_answer else ''
        return f"{context}\n{question}\n{cstr}\n答案：{answer}"
        
    def build_prompt(self, task_name, data, dataset, shot):
        prompt = ''
        if shot == 0:
            prompt += self.build_an_example_prompt(data, with_answer=False)
        else:
            few_shot_data = self.few_shot_data
            for i in range(min(len(few_shot_data), shot)):
                prompt += self.build_an_example_prompt(few_shot_data[i], with_answer=True) + '\n'
            prompt += self.build_an_example_prompt(data, with_answer=False)
        return prompt
    

class ItemOrder(BaseEval):
    def get_task_names(self):
        task_names = [f'item_num_{x}' for x in range(3, 10)]
        return task_names
    
    def loaddataset(self, task_name, *args):
        val = json.load(open(
            os.path.join(self.data_path, 'val.json'), encoding='utf-8'))
        test = json.load(open(
            os.path.join(self.data_path, 'test.json'), encoding='utf-8'))        
        item_num = int(task_name.replace('item_num_', ''))
        test = list(filter(lambda x: x['item_num'] == item_num,  test))
        dataset = datasets.DatasetDict()
        dataset['val'] = datasets.Dataset.from_list(val)
        dataset['test'] = datasets.Dataset.from_list(test)

        return dataset
    
    def get_answer(self, data):
        return data['answer'] if 'answer' in data else 'NoAnswer'   
    
    def set_fewshot_dataset(self, dataset):
        self.few_shot_data = dataset['val'].shuffle(seed=random_seed) 

    def build_an_example_prompt(self, data, with_answer=True, flag=0):
        question = data['question']
        choice = [f'{a}：{b}' for a, b in zip(self.candidate_labels, data['candidates'])]
        cstr = '\n'.join(choice)
        answer = data['answer'] if with_answer else ''
        return f"{question}\n{cstr}\n答案：{answer}"
        
    def build_prompt(self, task_name, data, dataset, shot):
        prompt = ''
        if shot == 0:
            prompt += self.build_an_example_prompt(data, with_answer=False)
        else:
            few_shot_data = self.few_shot_data
            for i in range(min(len(few_shot_data), shot)):
                prompt += self.build_an_example_prompt(few_shot_data[i], with_answer=True) + '\n'
            prompt += self.build_an_example_prompt(data, with_answer=False)
        return prompt


class TimeOrder(BaseEval):
    def get_task_names(self):
        task_names = ['timeorder']
        return task_names
    
    def loaddataset(self, task_name, *args):
        val = json.load(open(
            os.path.join(self.data_path, 'val.json'), encoding='utf-8'))
        test = json.load(open(
            os.path.join(self.data_path, 'test.json'), encoding='utf-8'))   
        dataset = datasets.DatasetDict()
        dataset['val'] = datasets.Dataset.from_list(val)
        dataset['test'] = datasets.Dataset.from_list(test)

        return dataset
    
    def get_answer(self, data):
        return data['answer'] if 'answer' in data else 'NoAnswer'   
    
    def set_fewshot_dataset(self, dataset):
        self.few_shot_data = dataset['val'].shuffle(seed=random_seed) 

    def build_an_example_prompt(self, data, with_answer=True, flag=0):
        prompt = data['system_prompt']
        question = data['question']
        context = data['context']
        choice = [f'{a}：{b}' for a, b in zip(self.candidate_labels, data['candidates'])]
        cstr = '\n'.join(choice)
        answer = data['answer'] if with_answer else ''
        return f"{prompt}\n{context}\n{question}\n{cstr}\n答案：{answer}"
        
    def build_prompt(self, task_name, data, dataset, shot):
        prompt = ''
        if shot == 0:
            prompt += self.build_an_example_prompt(data, with_answer=False)
        else:
            few_shot_data = self.few_shot_data
            for i in range(min(len(few_shot_data), shot)):
                prompt += self.build_an_example_prompt(few_shot_data[i], with_answer=True) + '\n'
            prompt += self.build_an_example_prompt(data, with_answer=False)
        return prompt
    

class CEval(BaseEval):
    task2desc = utils2.CEvalTasks

    def get_task_names(self):
        task_names = self.task2desc.keys()
        task_names = list(sorted(task_names))
        return task_names
     
    def get_answer(self, data):
        return data['answer'] if 'answer' in data else 'NoAnswer'   
    
    def set_fewshot_dataset(self, dataset):
        self.few_shot_data = dataset['dev'].shuffle(seed=random_seed) 

    def build_an_example_prompt(self, data, with_answer=True, flag=0):
        question = data['question']
        choice = [
                "A: " + data["A"],
                "B: " + data["B"],
                "C: " + data["C"],
                "D: " + data["D"],]
        
        cstr = '\n'.join(choice)
        answer = data['answer'] if with_answer else ''
        if flag == 0:
            return f"{question}\n{cstr}\n答案：{answer}"
        if flag == 'chat':
            return f"### Human:{question}\n{cstr}\n### Assistant:答案：{answer}"
        elif flag == 'bn':
            return f"<reserved_2>{question}\n{cstr}\n<reserved_3>答案：{answer}"
        
    def build_prompt(self, task_name, data, dataset, shot):
        model_id = self.model_id.split('/')[-1]
        if model_id == 'bc-type3zh-sft':
            flag = 'chat'
        elif model_id == 'bc-type3zh-bnsft' or model_id == 'bc-type3zh-bntype_sft':
            flag = 'bn'
        else:
            flag = 0

        if shot == 0:
            prompt = f"以下是中国关于{self.task2desc[task_name]}考试的单项选择题，请选出其中的正确答案。\n\n{self.build_an_example_prompt(data, with_answer=False, flag=flag)}"
        else:
            few_shot_data = self.few_shot_data
            prompt = f"以下是中国关于{self.task2desc[task_name]}考试的单项选择题，请选出其中的正确答案。\n"
            for i in range(min(len(few_shot_data), shot)):
                prompt += '\n' + self.build_an_example_prompt(few_shot_data[i], with_answer=True, flag=flag)
            prompt += '\n' + self.build_an_example_prompt(data, with_answer=False, flag=flag)
        return prompt


class FinancelQ(BaseEval):
    def get_task_names(self):
        subjects = utils2.FinancelQSubjects
        return subjects
    
    def loaddataset(self, task_name, *args):
        filename = f'{task_name}.csv'
        mydataset = datasets.load_dataset('csv', data_files={split: os.path.join(self.data_path, split, filename) for split in ('test', 'dev')})
        return mydataset
    
    def get_answer(self, data):
        return data['Answer'] if 'Answer' in data else 'NoAnswer'   
    
    def set_fewshot_dataset(self, dataset):
        self.few_shot_data = dataset['dev'].shuffle(seed=random_seed) 

    def build_an_example_prompt(self, data, with_answer=True, flag=0):
        question = data['Question']
        choice = [
                "A. " + data["A"],
                "B. " + data["B"],
                "C. " + data["C"],
                "D. " + data["D"],]
        
        cstr = '\n'.join(choice)
        answer = self.get_answer(data) if with_answer else ''
        return f"题目：{question}\n{cstr}\n答案是：{answer}\n"
        
    def build_prompt(self, task_name, data, dataset, shot):
        model_id = self.model_id.split('/')[-1]
        if model_id == 'bc-type3zh-sft':
            flag = 'chat'
        elif model_id == 'bc-type3zh-bnsft' or model_id == 'bc-type3zh-bntype_sft':
            flag = 'bn'
        else:
            flag = 0

        if shot == 0:
            prompt = f"以下是关于{task_name}的单项选择题，请直接给出正确答案的选项。\n\n{self.build_an_example_prompt(data, with_answer=False, flag=flag)}"
        else:
            few_shot_data = self.few_shot_data
            prompt = f"以下是关于{task_name}的单项选择题，请直接给出正确答案的选项。\n"
            for i in range(min(len(few_shot_data), shot)):
                prompt += '\n' + self.build_an_example_prompt(few_shot_data[i], with_answer=True, flag=flag)
            prompt += '\n' + self.build_an_example_prompt(data, with_answer=False, flag=flag)
        return prompt
    

class Gaokao(BaseEval):
    name_map = {
        'Biology': '2010-2022_Biology_MCQs.jsonl',
        'Physics': '2010-2022_Physics_MCQs.jsonl',
        'Political': '2010-2022_Political_Science_MCQs.jsonl',
        'History': '2010-2022_History_MCQs.jsonl',
        'Math_I': '2010-2022_Math_I_MCQs.jsonl',
        'Chemistry': '2010-2022_Chemistry_MCQs.jsonl',
        'English': '2010-2013_English_MCQs.jsonl',
        'Math_II': '2010-2022_Math_II_MCQs.jsonl',
        'Chinese': '2010-2022_Chinese_Lang_and_Usage_MCQs.jsonl',
    }
    
    def get_answer(self, data):
        return data['answer'] if 'answer' in data else 'NoAnswer'   
    
    def loaddataset(self, task_name, *args):
        filename = self.name_map[task_name]
        mydataset = datasets.load_dataset('json', data_files={split: os.path.join(self.data_path, split, filename) for split in ('train', 'dev')})
        return mydataset
    
    def get_task_names(self):
        task_names = []
        print(self.data_path)
        task_names = list(sorted(self.name_map.keys()))
        print(task_names)
        return task_names
    
    def build_an_example_prompt(self, data, with_answer=True, flag=0):
        question = data['question']
        choice = [
                "A: " + data["A"],
                "B: " + data["B"],
                "C: " + data["C"],
                "D: " + data["D"],]
        
        cstr = '\n'.join(choice)
        answer = data['answer'] if with_answer else ''
        if flag == 0:
            return f"{question}\n{cstr}\n答案：{answer}"
        if flag == 'chat':
            return f"### Human:{question}\n{cstr}\n### Assistant:答案：{answer}"
        elif flag == 'bn':
            return f"<reserved_2>{question}\n{cstr}\n<reserved_3>答案：{answer}"
        
    def build_prompt(self, task_name, data, dataset, shot):
        model_id = self.model_id.split('/')[-1]
        if model_id == 'bc-type3zh-sft':
            flag = 'chat'
        elif model_id == 'bc-type3zh-bnsft' or model_id == 'bc-type3zh-bntype_sft':
            flag = 'bn'
        else:
            flag = 0

        if shot == 0:
            prompt = f"{self.build_an_example_prompt(data, with_answer=False, flag=flag)}"
        else:
            shuffle_train = self.few_shot_data
            prompt = ""
            for i in range(min(len(shuffle_train), shot)):
                prompt += '\n' + self.build_an_example_prompt(shuffle_train[i], with_answer=True, flag=flag)
            prompt += '\n' + self.build_an_example_prompt(data, with_answer=False, flag=flag)
        return prompt
    
   
class AGIEval(BaseEval):
    
    name_map = {
        'logiqa-zh': 'logiqa-zh.jsonl',
        # 'aqua-rat': 'aqua-rat.jsonl',
        # 'lsat-lr': 'lsat-lr.jsonl',
        # 'lsat-ar': 'lsat-ar.jsonl',
        # 'lsat-rc': 'lsat-rc.jsonl',
        'gaokao-chinese': 'gaokao-chinese.jsonl',
        'gaokao-english': 'gaokao-english.jsonl',
        'sat-math': 'sat-math.jsonl',
        'gaokao-history': 'gaokao-history.jsonl',
        'sat-en-without-passage': 'sat-en-without-passage.jsonl',
        'gaokao-physics': 'gaokao-physics.jsonl',
        'gaokao-biology': 'gaokao-biology.jsonl',
        'sat-en': 'sat-en.jsonl',
        'gaokao-chemistry': 'gaokao-chemistry.jsonl',
        'gaokao-mathqa': 'gaokao-mathqa.jsonl',
        'gaokao-geography': 'gaokao-geography.jsonl',
        'logiqa-en': 'logiqa-en.jsonl',
    }

    def get_answer(self, data):
        return data['answer'] if 'answer' in data else 'NoAnswer'    

    def get_task_names(self):
        task_names = []
        print(self.data_path)
        task_names = list(sorted(self.name_map.keys()))
        return task_names
    
    def loaddataset(self, task_name, *args):
        filename = self.name_map[task_name]
        mydataset = datasets.load_dataset('json', data_files={split: os.path.join(self.data_path, split, filename) for split in ('train', 'dev')})
        return mydataset
    
    def build_an_example_prompt(self, data, with_answer=True, flag=0):
        question = data['question']
        choice = [
                "A: " + data["A"],
                "B: " + data["B"],
                "C: " + data["C"],
                "D: " + data["D"],]
        cstr = '\n'.join(choice)
        answer = data['answer'] if with_answer else ''
        if flag == 0:
            return f"{question}\n{cstr}\n答案：{answer}"
        if flag == 'chat':
            return f"### Human:{question}\n{cstr}\n### Assistant:答案：{answer}"
        elif flag == 'bn':
            return f"<reserved_2>{question}\n{cstr}\n<reserved_3>答案：{answer}"
        
    def build_prompt(self, task_name, data, dataset, shot):
        model_id = self.model_id.split('/')[-1]
        if model_id == 'bc-type3zh-sft':
            flag = 'chat'
        elif model_id == 'bc-type3zh-bnsft' or model_id == 'bc-type3zh-bntype_sft':
            flag = 'bn'
        else:
            flag = 0

        if shot == 0:
            prompt = f"{self.build_an_example_prompt(data, with_answer=False, flag=flag)}"
        else:
            shuffle_train = self.few_shot_data
            prompt = ""
            for i in range(min(len(shuffle_train), shot)):
                prompt += '\n' + self.build_an_example_prompt(shuffle_train[i], with_answer=True, flag=flag)
            prompt += '\n' + self.build_an_example_prompt(data, with_answer=False, flag=flag)
        return prompt
    

class MedQA(BaseEval):
    def get_task_names(self):
        return ['usmle', 'mcmle']
    
    def get_answer(self, data):
        return self.candidate_labels[data['choices'].index(data['answer'][0])] if 'answer' in data else 'NoAnswer'    
    
    def loaddataset(self, task_name, *args):
        task_name = os.path.join(self.data_path, task_name)
        print(task_name)
        return datasets.load_dataset('json', data_files={'test': os.path.join(task_name, 'test.jsonl'),
                                                         'train': os.path.join(task_name, 'train.jsonl'),
                                                         'validation': os.path.join(task_name, 'validation.jsonl')})

    def build_an_example_prompt(self, data, with_answer=True):
        question = data['question']
        choice = [f'{label}: {v}' for label, v in zip(self.candidate_labels, data['choices'])]
        cstr = '\n'.join(choice)
        answer = self.candidate_labels[data['choices'].index(data['answer'][0])] if with_answer else ''
        return f"{question}\n{cstr}\nAnswer: {answer}"
        
    def build_prompt(self, task_name, data, dataset, shot):
        if shot == 0:
            prompt = f"{self.build_an_example_prompt(data, with_answer=False)}"
        else:
            few_shot_data = self.few_shot_data
            prompt = ""
            for i in range(min(len(few_shot_data), shot)):
                prompt += '\n' + self.build_an_example_prompt(few_shot_data[i], with_answer=True)
            prompt += '\n' + self.build_an_example_prompt(data, with_answer=False)
        return prompt


class SuperClue0801(BaseEval):
    def get_task_names(self):
        return ['']
    
    def get_answer(self, data):
        return data['answer'] if 'answer' in data else 'NoAnswer'   

    def build_an_example_prompt(self, data, with_answer=True):
        question = data['question']
        choice = json.loads(data['choices'].replace("'", '"'))
        cstr = '\n'.join(choice)
        answer = self.candidate_labels[data['choices'].index(data['answer'][0])] if with_answer else ''
        return f"{question}\n{cstr}\nAnswer: {answer}"
        
    def build_prompt(self, task_name, data, dataset, shot):
        if shot == 0:
            prompt = f"{self.build_an_example_prompt(data, with_answer=False)}"
        else:
            few_shot_data = self.few_shot_data
            prompt = ""
            for i in range(min(len(few_shot_data), shot)):
                prompt += '\n' + self.build_an_example_prompt(few_shot_data[i], with_answer=True)
            prompt += '\n' + self.build_an_example_prompt(data, with_answer=False)
        return prompt


class MMLU(BaseEval):
    def get_task_names(self):
        subjects = list(utils2.MMLUSubcategories.keys())
        return subjects
    
    def get_answer(self, data):
        return data['answer'] if 'answer' in data else 'N'   

    def set_fewshot_dataset(self, dataset):
        self.few_shot_data = dataset['dev']
    
    def loaddataset(self, task_name, *args):
        print(task_name)
        names = ['question', 'A', 'B', 'C', 'D', 'answer']
        df1 = pd.read_csv(os.path.join(self.data_path, 'test', task_name + '_test.csv'), header=None, names=names)
        df2 = pd.read_csv(os.path.join(self.data_path, "dev", task_name + '_dev.csv'), header=None, names=names)
        mydataset = datasets.DatasetDict({
            'test': datasets.Dataset.from_pandas(df1),
            'dev': datasets.Dataset.from_pandas(df2),
        })
        return mydataset

    def build_an_example_prompt(self, data, with_answer=True):
        question = data['question']
        choice = [f'{label}: {data[label]}' for label in self.candidate_labels]
        cstr = '\n'.join(choice)
        answer = self.get_answer(data) if with_answer else ''
        return f"{question}\n{cstr}\nAnswer:{answer}"
        
    def build_prompt(self, task_name, data, dataset, shot):
        prompt = f"The following are multiple choice questions (with answers) about {task_name.replace('_', ' ')}.\n\n"
        if shot == 0:
            prompt += f"{self.build_an_example_prompt(data, with_answer=False)}"
        else:
            few_shot_data = self.few_shot_data
            for i in range(min(len(few_shot_data), shot)):
                prompt += self.build_an_example_prompt(few_shot_data[i], with_answer=True) + '\n\n'
            prompt += self.build_an_example_prompt(data, with_answer=False)
        return prompt
    

class CMMLU(BaseEval):
    def get_task_names(self):
        subjects = list(utils2.CMMLUTasks.keys())
        return subjects
    
    def get_answer(self, data):
        return data['answer'] if 'answer' in data else 'N'    
    
    def set_fewshot_dataset(self, dataset):
        self.few_shot_data = dataset['dev']
    
    def loaddataset(self, task_name, *args):
        print(task_name)
        names = ['id', 'question', 'A', 'B', 'C', 'D', 'answer']
        df1 = pd.read_csv(os.path.join(self.data_path, 'test', task_name + '.csv'), header=0, names=names)
        df2 = pd.read_csv(os.path.join(self.data_path, "dev", task_name + '.csv'), header=0, names=names)
        mydataset = datasets.DatasetDict({
            'test': datasets.Dataset.from_pandas(df1),
            'dev': datasets.Dataset.from_pandas(df2),
        })
        return mydataset

    def build_an_example_prompt(self, data, with_answer=True):
        question = data['question']
        choice = [f'{label}. {data[label]}' for label in self.candidate_labels]
        cstr = '\n'.join(choice)
        answer = self.get_answer(data) if with_answer else ''
        return f"题目：{question}\n{cstr}\n答案是：{answer}"
        
    def build_prompt(self, task_name, data, dataset, shot):
        prompt = f"以下是关于{utils2.CMMLUTasks[task_name]}的单项选择题，请直接给出正确答案的选项。\n\n"
        if shot == 0:
            prompt += f"{self.build_an_example_prompt(data, with_answer=False)}"
        else:
            few_shot_data = self.few_shot_data
            for i in range(min(len(few_shot_data), shot)):
                prompt += self.build_an_example_prompt(few_shot_data[i], with_answer=True) + '\n\n'
            prompt += self.build_an_example_prompt(data, with_answer=False)
        return prompt


class LongBench():        
    def __init__(self, model_id, data_path, output_dir, max_length):
        self.model_id = model_id
        self.data_path = data_path
        self.output_dir = output_dir
        self.max_length = max_length

    def set_model_and_tokenizer(self, model, tokenizer):
        self.model = model
        self.tokenizer=tokenizer

    def run(self):
        model_id = self.model_id
        data_path = self.data_path
        output_dir = self.output_dir
        max_length = self.max_length
        model, tokenizer = self.model, self.tokenizer

        from longbench import pred as longbench_pred
        from collections import namedtuple
        Args = namedtuple('Args' , 'e model')
        args = Args(False, model_id)

        longbench_pred.seed_everything(42)
        model2path = json.load(open("longbench/config/model2path.json", "r"))
        model2maxlen = json.load(open("longbench/config/model2maxlen.json", "r"))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_name = utils2.longbench_model_to_max_length_k(model_id, max_length)
        # define your model
        # model, tokenizer = load_model_and_tokenizer(model_id, model_name, device)
        # max_length = model2maxlen[model_name]
        if args.e:
            longbench_datasets = ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report", "multi_news", \
                "trec", "triviaqa", "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]
        else:
            longbench_datasets = ["narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "hotpotqa", "2wikimqa", "musique", \
                        "dureader", "gov_report", "qmsum", "multi_news", "vcsum", "trec", "triviaqa", "samsum", "lsht", \
                        "passage_count", "passage_retrieval_en", "passage_retrieval_zh", "lcc", "repobench-p"]
        # we design specific prompt format and max generation length for each task, feel free to modify them to optimize model output
        dataset2prompt = json.load(open("longbench/config/dataset2prompt.json", "r"))
        dataset2maxlen = json.load(open("longbench/config/dataset2maxlen.json", "r"))
        # predict on each dataset
        if args.e:
            tail = '_e'
        else:
            tail = ''
        output_dir = os.path.join(output_dir, 'LongBench', f"pred{tail}/{model_name}")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        for dataset in longbench_datasets:
            filename = os.path.join(data_path, dataset+tail+'.jsonl')
            data = datasets.load_dataset('json', data_files={'test': filename})
            out_path = os.path.join(output_dir, f"{dataset}.jsonl")

            prompt_format = dataset2prompt[dataset]
            max_gen = dataset2maxlen[dataset]
            data = data['test']
            preds = longbench_pred.get_pred(model, tokenizer, data, max_length, max_gen, prompt_format, dataset, device, model_name)
            with open(out_path, "w", encoding="utf-8") as f:
                for pred in preds:
                    json.dump(pred, f, ensure_ascii=False)
                    f.write('\n')


class JecQA(BaseEval):
    def get_task_names(self):
        task_names = utils2.JecQATasks.keys()
        task_names = list(sorted(task_names))
        return task_names
    
    def set_fewshot_dataset(self, dataset):
        self.few_shot_data = dataset['val']

    def loaddataset(self, task_name, *args):
        filename = os.path.join(self.data_path, task_name+'.json')
        print(filename)
        with open(filename, 'r', encoding='utf-8') as f:
            raw = f.readlines()
        origin_data = [json.loads(x) for x in raw]
        one_answer_data = list(filter(lambda x: len(x['answer']) == 1, origin_data))
        random.shuffle(one_answer_data)
        val_data = one_answer_data[: 10]
        test_data = one_answer_data[10:]
        mydataset = datasets.DatasetDict({
            'test': datasets.Dataset.from_list(test_data),
            'val': datasets.Dataset.from_list(val_data),
        })
        return mydataset
    
    def get_answer(self, data):
        return data['answer'][0] if 'answer' in data else 'NoAnswer'   
         
    def build_an_example_prompt(self, data, with_answer=True, flag=0):
        question = data['statement']
        choice = [
                "A: " + data['option_list']["A"],
                "B: " + data['option_list']["B"],
                "C: " + data['option_list']["C"],
                "D: " + data['option_list']["D"],]
        
        cstr = '\n'.join(choice)
        answer = data['answer'][0] if with_answer else ''
        if flag == 0:
            return f"{question}\n{cstr}\n答案：{answer}"
        if flag == 'chat':
            return f"### Human:{question}\n{cstr}\n### Assistant:答案：{answer}"
        elif flag == 'bn':
            return f"<reserved_2>{question}\n{cstr}\n<reserved_3>答案：{answer}"
        
    def build_prompt(self, task_name, data, dataset, shot):
        model_id = self.model_id.split('/')[-1]
        if model_id == 'bc-type3zh-sft':
            flag = 'chat'
        elif model_id == 'bc-type3zh-bnsft' or model_id == 'bc-type3zh-bntype_sft':
            flag = 'bn'
        else:
            flag = 0

        if shot == 0:
            prompt = f"以下是中国司法考试的单项选择题，请选出其中的正确答案。\n\n{self.build_an_example_prompt(data, with_answer=False, flag=flag)}"
        else:
            prompt = f"以下是中国司法考试的单项选择题，请选出其中的正确答案。\n"
            for i in range(min(len(self.few_shot_data), shot)):
                prompt += '\n' + self.build_an_example_prompt(self.few_shot_data[i], with_answer=True, flag=flag)
            prompt += '\n' + self.build_an_example_prompt(data, with_answer=False, flag=flag)
        return prompt
    

class Words2SentenceGenerate:
    def __init__(self, model_id, model, tokenizer, model_config, data_filename, output_dir, shot):
        self.data_filename = data_filename
        self.shot = shot
        self.model = model
        self.tokenizer = tokenizer
        self.model_config = model_config
        self.output_dir = output_dir
        self.output_dir2 = os.path.join(output_dir, self.__class__.__name__)
        if not os.path.exists(self.output_dir2):
            os.makedirs(self.output_dir2)
        self.filename_pfx = f"{model_id.split('/')[-1]}_{self.__class__.__name__}_{shot}-shot"

    def build_an_example_prompt(self, words, language, with_answer=True):
        if with_answer:
            answer = ''.join(words) if language == 'zh' else ' '.join(words) 
        else:
            answer = ''
        random.shuffle(words)
        prompt = ' '.join(words) 
        return f'词语：{prompt}\n句子：{answer}' if language == 'zh' else f'Words List: {prompt}\nSentence: {answer}' 
    
    def build_prompt(self, words, language, shot=0):
        assert type(words) == list
        if language == 'zh':
            prompt = '把下列词语整理成一句通顺的话：'
            # prompt = 'Organize the following words into a coherent sentence:'
            samples = [
                '我家 门口 有 一棵 小树',
                '操场 上 的 小草 发芽 了',
                '大雁 秋天 要 飞到 哪里去',
                '今天 阳光 明媚',
                '创造 属于 我们 的 故事'
            ]
        else:
            prompt = 'Organize the following words into a coherent sentence:'
            samples = [
                'There is a tree infront of my house',
                'The sun is shining brightly',
                'I have a cat named Tom',
                'They usually go for a walk every morning',
                'He enjoys playing video games during his spare time'
            ]
        for i in range(min(len(samples), shot)):
            prompt += '\n' + self.build_an_example_prompt(samples[i].split(' '), language, with_answer=True)
        prompt += '\n' + self.build_an_example_prompt(words, language, with_answer=False)
        return prompt
    
    @torch.no_grad()
    def predict_instance(self, prompt):
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").cuda()
        context_length = input_ids.shape[-1]
        output = self.model.generate(
                input_ids,
                max_new_tokens=self.model_config['max_new_tokens'],
                temperature=self.model_config['temperature'],
                top_p=self.model_config['top_p'],
                repetition_penalty=self.model_config['repetition_penalty'],
            )[0]
        pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
        return pred
    
    def save_results(self, record):
        filename = os.path.join(self.output_dir, 'all_results.json')
        print(filename)
        print(record)
        if os.path.exists(filename):
            data = json.load(open(filename, 'r'))
        else:
            data = dict()
        data.update(record)
        json.dump(data, open(filename, 'w'), indent=True, ensure_ascii=False)
        print(f'all results in: {filename}')
    
    def run(self):
        stime = get_time()
        details = []
        scores = []
        language2average = dict()
        for language in ('zh', 'en'):
            st = False
            with open(self.data_filename) as f:
                for line in f:
                    if '#' == line[0] and line.split('#')[-1].strip() == language:
                        lines = []
                        st = True
                        continue
                    if '#' == line[0] and line.split('#')[-1].strip() != language:
                        st = False
                    if st:
                        lines.append(line.strip())
            all_words = [list(filter(lambda x: len(x) > 0, x.split(' '))) for x in lines]
            subscores = []
            for i, words in enumerate(all_words):
                answer = ''.join(words) if language == 'zh' else ' '.join(words)
                prompt = self.build_prompt(words, language, shot=self.shot)
                full_pred = self.predict_instance(prompt)
                pred = full_pred.strip().split('\n')[0]
                lcs = utils2.longestCommonSubsequence(answer, pred)
                score = 100 * lcs / len(answer)
                scores.append(score)
                subscores.append(score)
                details.append({'id': i, 'words': words, 'pred': pred, 'answer': answer, 'lcs': lcs, 'score': f'{score:.2f}', 'prompt': prompt, 'full_pred': full_pred})
            language2average[language] =  f'{np.mean(subscores):.2f}'
        etime = get_time()
        record = {'start_time': stime, 'end_time': etime, 'average': f'{np.mean(scores):.2f}'}
        record.update(language2average)
        self.save_results({self.filename_pfx: record})
        with open(os.path.join(self.output_dir2, self.filename_pfx+'.jl'), 'w') as f:
            for x in details:
                f.write(json.dumps(x, ensure_ascii=False))
                f.write('\n')
        
        

class Linguistic(BaseEval):
    def get_task_names(self):
        return ['homophone', 'synonym', 'words2sentence_zh', 'words2sentence_en']
  
    def get_answer(self, data):
        return data['answer'] if 'answer' in data else 'NoAnswer' 
    
    def loaddataset(self, task_name, *args):
        task_name = os.path.join(self.data_path, task_name)
        print(task_name)
        return datasets.load_dataset(task_name)

    def build_an_example_prompt(self, data, with_answer=True):
        question = data['question']
        cstr = '\n'.join(data['choices'])
        answer = data['answer'] if with_answer else ''
        return f"{question}\n{cstr}\nAnswer: {answer}"
        
    def build_prompt(self, task_name, data, dataset, shot):
        if shot == 0:
            prompt = '\n' + f"{self.build_an_example_prompt(data, with_answer=False)}"
        else:
            prompt = ''
            few_shot_data = self.few_shot_data
            for i in range(min(len(few_shot_data), shot)):
                prompt += '\n' + self.build_an_example_prompt(few_shot_data[i], with_answer=True)
            prompt += '\n' + self.build_an_example_prompt(data, with_answer=False)
        return prompt


class QuaternionGen(BaseEval):
    def get_task_names(self):
        filename = os.path.join(self.data_path, 'quaternions3.json')
        data = json.load(open(filename))
        task_names = list(data.keys())
        return task_names
    
    def loaddataset(self, task_name, *args):
        print(task_name)
        filename = os.path.join(self.data_path, 'quaternions3.json')
        data = json.load(open(filename))
        data = data[task_name]
        dataset = datasets.DatasetDict({
            'test': datasets.Dataset.from_list(data),
        })
        return dataset
    
    def get_answer(self, data):
        return data['answer'] if 'answer' in data else 'NoAnswer'   
    
    def set_fewshot_dataset(self, dataset):
        self.few_shot_data = None

    def build_an_example_prompt(self, data, with_answer=True, flag=0):
        pass
        
    def build_prompt(self, task_name, data, dataset, shot):
        prompt = f"Please calculate the following expression:\n"
        examples = [
            'Expression: 132+19\nThe answer is: <<<151>>>',
            'Expression: 2+37+9\nThe answer is: <<<48>>>',
            'Expression: 298-19-182\nThe answer is: <<<97>>>',
            'Expression: 382-32-279\nThe answer is: <<<71>>>',
            'Expression: 52*71\nThe answer is: <<<3692>>>',
            'Expression: 82*47*2\nThe answer is: <<<7708>>>',
            'Expression: 1568/7\nThe answer is: <<<224>>>',
            'Expression: 2928/12/61\nThe answer is: <<<4>>>',
        ]
        prompt += '\n'.join(examples)
        prompt += f"\nExpression: {data['expression']}\nThe answer is: "
        return prompt

    def parser_generation(self, continuation):
        continuation = continuation[0]
        for seg in continuation.split('\n'):
            seg = seg.strip()
            if len(seg) > 0:
                seg = seg.replace('The answer is: <<<', '').replace('>', '').replace('<', '')
                try:
                    pred = str(int(float(seg)))
                except:
                    pred = seg
                return pred
        return continuation
    
    def is_correct(self, pred, answer):
        return answer == pred 


class QuaternionGenMixOp(QuaternionGen):
    def get_task_names(self):
        filename = os.path.join(self.data_path, 'quaternions_mix_operates2.json')
        data = json.load(open(filename))
        task_names = list(data.keys())
        return task_names 

    def loaddataset(self, task_name, *args):
        print(task_name)
        filename = os.path.join(self.data_path, 'quaternions_mix_operates2.json')
        data = json.load(open(filename))
        data = data[task_name]
        dataset = datasets.DatasetDict({
            'test': datasets.Dataset.from_list(data),
        })
        return dataset
    
    def build_prompt(self, task_name, data, dataset, shot):
        prompt = f"Please calculate the following expression:\n"
        examples = [
            '132+19=151',
            '2+37+9=48',
            '298-19-182=97',
            '382-32-279=71',
            '52*71=3692',
            '82*47*2=7708',
            '1568/7=224',
            '2928/12/61=4',
        ]
        prompt += '\n'.join(examples)
        prompt += f"\n{data['expression']}="
        return prompt


class QuaternionMC(BaseEval):
    def get_task_names(self):
        return ['ADD', 'MINUS', 'TIMES', 'DIV']
  
    def get_answer(self, data):
        return data['answer'] if 'answer' in data else 'NoAnswer' 
    
    def loaddataset(self, task_name, *args):
        task_name = os.path.join(self.data_path, task_name)
        print(task_name)
        return datasets.load_dataset(task_name)

    def build_an_example_prompt(self, data, with_answer=True):
        question = data['question']
        cstr = '\n'.join(data['choices'])
        answer = data['answer'] if with_answer else ''
        return f"{question}\n{cstr}\nAnswer: {answer}"
        
    def build_prompt(self, task_name, data, dataset, shot):
        if shot == 0:
            prompt = '计算下列式子的结果，并选择正确的选项：\n' + f"{self.build_an_example_prompt(data, with_answer=False)}"
        else:
            prompt = '计算下列式子的结果，并选择正确的选项：'
            few_shot_data = self.few_shot_data
            for i in range(min(len(few_shot_data), shot)):
                prompt += '\n' + self.build_an_example_prompt(few_shot_data[i], with_answer=True)
            prompt += '\n' + self.build_an_example_prompt(data, with_answer=False)
        return prompt    
    

class PubMedQA(Linguistic):
    def get_task_names(self):
        return ['']

class PubMedQA_V2(Linguistic):
    def get_task_names(self):
        return ['']

class MedMCQA(Linguistic):
    def get_task_names(self):
        return ['']
    
 
class MedExam(Linguistic):
    def get_task_names(self):
        return ['']       
    

class CMExam(BaseEval):
    def get_task_names(self):
        return ['']
  
    def get_answer(self, data):
        return data.get('Answer', 'NoAnswer')
    
    def loaddataset(self, task_name, *args):
        df1 = pd.read_csv(os.path.join(self.data_path, 'test_with_annotations.csv'), header=0)
        df2 = pd.read_csv(os.path.join(self.data_path, 'val.csv'), header=0)
        df3 = pd.read_csv(os.path.join(self.data_path, 'train.csv'), header=0)
        cmexam_dataset = datasets.DatasetDict({
            'test': datasets.Dataset.from_pandas(df1.iloc[:, :4]),
            'val': datasets.Dataset.from_pandas(df2),
            'train': datasets.Dataset.from_pandas(df3),
        })
        cmexam_dataset = datasets.DatasetDict({split: 
                      cmexam_dataset[split].filter(lambda x: x['Answer'] is not None and len(x['Answer']) == 1) 
                      for split in ('test', 'train', 'val')})
        return cmexam_dataset

    def build_an_example_prompt(self, data, with_answer=True):
        question = data['Question']
        cstr = data['Options']
        answer = self.get_answer(data) if with_answer else ''
        return f"{question}\n{cstr}\n答案: {answer}"
        
    def build_prompt(self, task_name, data, dataset, shot):
        prompt = '假设你是一位医疗行业专家，请回答下列问题，注意是单选题，只需要返回一个最合适的选项。\n\n'
        if shot == 0:
            prompt = f"{self.build_an_example_prompt(data, with_answer=False)}"
        else:
            few_shot_data = self.few_shot_data
            for i in range(min(len(few_shot_data), shot)):
                prompt += self.build_an_example_prompt(few_shot_data[i], with_answer=True) + '\n\n'
            prompt += self.build_an_example_prompt(data, with_answer=False)+ '\n'
        return prompt


class BCEvalZh(BaseEval):
    def get_task_names(self):   
        val = [json.loads(x) for x in open(os.path.join(self.data_path, 'val.jsonl'))]
        subjects = list(sorted(set([x['subject'] for x in val])))             
        return subjects
  
    def get_answer(self, data):
        return data.get('answer', 'N')
    
    def set_fewshot_dataset(self, dataset):
        self.few_shot_data = dataset['val'].shuffle(seed=random_seed)
    
    def loaddataset(self, task_name, *args):
        val = [json.loads(x) for x in open(os.path.join(self.data_path, 'val.jsonl'))]
        test = [json.loads(x) for x in open(os.path.join(self.data_path, 'test.jsonl'))]

        val = list(filter(lambda x: x['subject'] == task_name,  val))
        test = list(filter(lambda x: x['subject'] == task_name,  test))

        valp = pd.json_normalize(val)
        testp = pd.json_normalize(test)

        mydataset = datasets.DatasetDict({
            'test': datasets.Dataset.from_pandas(testp),
            'val': datasets.Dataset.from_pandas(valp),
        })

        shuffled_dataset = datasets.DatasetDict()
        for split in mydataset:
            new_dataset = []
            for data in mydataset[split]:
                order = ['A', 'B', 'C', 'D']
                new_order = ['A', 'B', 'C', 'D']
                random.shuffle(new_order)
                new_data = copy.deepcopy(data)
                new_answer = new_order[order.index(data['answer'])]
                new_data['answer'] = new_answer
                for o, no in zip(order, new_order):
                    new_data[no] = data[o]
                new_dataset.append(new_data)
            shuffled_dataset[split] = datasets.Dataset.from_list(new_dataset)       

        return mydataset

    def build_an_example_prompt(self, data, with_answer=True):
        question = data['question']
        choice = [
                "A: " + data["A"],
                "B: " + data["B"],
                "C: " + data["C"],
                "D: " + data["D"],]
        cstr = '\n'.join(choice)
        answer = self.get_answer(data) if with_answer else ''
        return f"{question}\n{cstr}\n答案: {answer}"
        
    def build_prompt(self, task_name, data, dataset, shot):
        prompt = f"以下是关于{data['ch_subject']}的单项选择题，请选出其中的正确答案。\n\n"
        if shot == 0:
            prompt = f"{self.build_an_example_prompt(data, with_answer=False)}"
        else:
            few_shot_data = self.few_shot_data
            for i in range(min(len(few_shot_data), shot)):
                prompt += self.build_an_example_prompt(few_shot_data[i], with_answer=True) + '\n\n'
            prompt += self.build_an_example_prompt(data, with_answer=False)+ '\n'
        return prompt


class LegalBench(BaseEval):
    def loaddataset(self, task_name, *args):
        print(self.data_path, task_name)
        return datasets.load_dataset(self.data_path, task_name)

    def run_my_task(self, task_name, shot, split='test'):
        dataset = self.loaddataset(task_name)
        # Load base prompt
        df = dataset[split].to_pandas()
        with open(f"legalbench/tasks/{task_name}/base_prompt.txt") as in_file:
            prompt_template = in_file.read()
        prompts = zxrlb_utils.generate_prompts(prompt_template=prompt_template, data_df=df)  
        generations = []
        for prompt in prompts:
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt").cuda()
            context_length = input_ids.shape[-1]
            output = self.model.generate(
                input_ids,
                max_new_tokens=self.max_gen,
                temperature=self.temperature,
                top_p=self.top_p,
                repetition_penalty=self.repetition_penalty,
            )[0]
            generate = self.tokenizer.decode(output[context_length:], skip_special_tokens=True)
            generations.append(generate)
        score = zxrlb_evaluation.evaluate(task_name, generations, df["answer"].tolist())
        return score, prompts, generations

    def run(self, shot, split):
        self.max_gen = 100
        self.temperature = 0.9
        self.top_p = 0.5
        self.repetition_penalty = 1.1
        self.filename_pfx = f"{self.model_id.split('/')[-1]}_{self.__class__.__name__}"
        out_path = os.path.join(self.output_dir, self.__class__.__name__, self.filename_pfx)
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        record = dict()
        for task in zxrlb_tasks.TASKS[:]:
            print(self.__class__.__name__, task)
            score, prompts, generations = self.run_my_task(task, None)
            record['start_time'] = self.start_time
            record['end_time'] = get_time()
            record[task] = score
            self.save_results({self.filename_pfx: record})

            filename = os.path.join(out_path, task + '.json')
            details = [{'prompt': p, 'generate': g} for p, g in zip(prompts, generations)]
            json.dump(details, open(filename, 'w', encoding='utf-8'), indent=True, ensure_ascii=False)
                
    def calc_acc(self):
        pass


class SelfBench(BaseEval):

    def get_task_names(self):
        task_names = ['dl', 'ls', 'lz', 'sxzz']
        task_names = list(sorted(task_names))
        return task_names

    def myread_file(self, task_name):
        data = []
        for subname in ('national_A', 'national_B'):
            if task_name == 'few_shot':
                filename = os.path.join(self.data_path, task_name + '.txt')
            else:
                filename = os.path.join(self.data_path, subname + '_' + task_name + '.txt')
            with open(filename, 'r', encoding='utf-8') as f:
                adata = []
                for line in f:
                    line = line.strip()
                    if len(line) == 0:
                        if len(adata) > 0:
                            assert len(adata) == 6, print(adata,  len(adata))
                            data.append(adata)
                            adata = []
                        continue
                    adata.append(line)
        data = [self.myparse_adata(x) for x in data]
        return data
    
    def myparse_adata(self, adata):
        segs = adata[0].strip().split('.')
        idx = segs[0]
        question = '.'.join(segs)
        choices = adata[1: 5]
        newcho = []
        for candidate in choices:
            segs = candidate.strip().split('.')
            label = segs[0].upper()
            candidate = '.'.join([label] + segs[1:])
            newcho.append(candidate)
        answer = adata[-1]
        return {'id': idx, 'question': question, 'choice': newcho, 'answer': answer.strip().upper()}

    def loaddataset(self, task_name, *args):
        test_data = self.myread_file(task_name)
        train_data = self.myread_file('few_shot')
        return {'train': train_data, 'test': test_data}
    
    def get_answer(self, data):
        return data['answer'] if 'answer' in data else 'NoAnswer'   

    def build_an_example_prompt(self, data, with_answer=True, flag=0):
        question = data['question']
        choice = data['choice']
        cstr = '\n'.join(choice)
        answer = data['answer'] if with_answer else ''
        if flag == 0:
            return f"{question}\n{cstr}\n答案：{answer}"
        if flag == 'chat':
            return f"### Human:{question}\n{cstr}\n### Assistant:答案：{answer}"
        elif flag == 'bn':
            return f"<reserved_2>{question}\n{cstr}\n<reserved_3>答案：{answer}"
        
    def build_prompt(self, task_name, data, dataset, shot):
        model_id = self.model_id.split('/')[-1]
        if model_id == 'bc-type3zh-sft':
            flag = 'chat'
        elif model_id == 'bc-type3zh-bnsft' or model_id == 'bc-type3zh-bntype_sft':
            flag = 'bn'
        else:
            flag = 0

        if shot == 0:
            prompt = f"{self.build_an_example_prompt(data, with_answer=False, flag=flag)}"
        else:
            few_shot_data = self.few_shot_data
            prompt = ""
            for i in range(min(len(few_shot_data), shot)):
                prompt += '\n' + self.build_an_example_prompt(few_shot_data[i], with_answer=True, flag=flag)
            prompt += '\n' + self.build_an_example_prompt(data, with_answer=False, flag=flag)
        return prompt


class GSM8K(BaseEval):
    gsm8k_evalutor = evaluator.GSM8KEvaluator()
    def get_task_names(self):
        return ["gsm8k"]
    
    def get_answer(self, data):
        return data['answer']

    def build_an_example_prompt(self, data, with_answer=True, flag=0):
        question = data['question']
        prompt_in = f"Question: {question}\nLet's think step by step\nAnswer:"

        if with_answer:
            answer = data['answer']
            prompt_out = f"\n{answer}"
            return prompt_in+prompt_out
        else:
            return prompt_in

    def set_fewshot_dataset(self, dataset):
        self.few_shot_data = json.load(open("few_shot.json", encoding='utf-8'))[self.get_task_names()[0]]['ocp']

    def build_prompt(self, task_name, data, dataset, shot):
        prompt = []
        for one_shot in self.few_shot_data:
            prompt.append(self.build_an_example_prompt(one_shot, with_answer=True))
        prompt.append(self.build_an_example_prompt(data, with_answer=False))
        prompt = "\n".join(prompt)
        return prompt

    def is_correct(self, prediction, reference):
        return self.gsm8k_evalutor.score(prediction, reference)
        
    def parser_generation(self, continuation):
        continuation = continuation[0]
        return continuation.lstrip()  # 部分模型（如SOLAR）最左边会生成两个回车符，评估代码默认用'\n\n'分割后的第一段文本，故导致评估结果偏低。此处去掉最左边的回车符以减少格式对结果的影响


class MATH(BaseEval):
    math_evaluator = evaluator.MATHEvaluator()
    def get_task_names(self):
        return ["math"]

    def get_answer(self, data):
        return data['solution']

    def set_fewshot_dataset(self, dataset):
        self.few_shot_data = json.load(open("few_shot.json", encoding='utf-8'))[self.get_task_names()[0]]['ocp']

    def build_an_example_prompt(self, data, with_answer=True, flag=0):
        problem = data['problem']
        prompt_in = f"Problem:\n{problem}\nSolution:\n"

        if with_answer:
            solution = data['solution']
            prompt_out = f"{solution}"
            return prompt_in+prompt_out
        else:
            return prompt_in

    def build_prompt(self, task_name, data, dataset, shot):
        prompt = []
        for one_shot in self.few_shot_data:
            # one_shot['problem'] = eval("'{}'".format(one_shot['problem']))
            # one_shot['solution'] = eval("'{}'".format(one_shot['solution']))
            prompt.append(self.build_an_example_prompt(one_shot, with_answer=True))
        prompt.append(self.build_an_example_prompt(data, with_answer=False))
        prompt = "\n".join(prompt)
        return prompt

    def is_correct(self, prediction, reference):
        return self.math_evaluator.score(prediction, reference)

    def loaddataset(self, task_name, *args):

        def remove_boxed(s):
            left = '\\boxed{'
            try:
                assert s[:len(left)] == left
                assert s[-1] == '}'
                return s[len(left):-1]
            except Exception:
                return None

        def last_boxed_only_string(string):
            idx = string.rfind('\\boxed')
            if idx < 0:
                idx = string.rfind('\\fbox')
                if idx < 0:
                    return None

            i = idx
            right_brace_idx = None
            num_left_braces_open = 0
            while i < len(string):
                if string[i] == '{':
                    num_left_braces_open += 1
                if string[i] == '}':
                    num_left_braces_open -= 1
                    if num_left_braces_open == 0:
                        right_brace_idx = i
                        break
                i += 1

            if right_brace_idx is None:
                retval = None
            else:
                retval = string[idx:right_brace_idx + 1]

            return retval

        dataset = DatasetDict()
        data = json.load(open(os.path.join(self.data_path, "math.json")))
        raw_data = []
        for i in data.keys():
            raw_data.append({
                'problem':
                data[i]['problem'],
                'solution':
                remove_boxed(last_boxed_only_string(data[i]['solution']))
            })
        dataset['test'] = Dataset.from_list(raw_data)
        dataset['test'] = dataset['test'].shuffle(random_seed).shard(num_shards=10, index=0)  # math数据量太大，有5000条，只测其中的十分之一
        return dataset
        
    def parser_generation(self, continuation):
        continuation = continuation[0]
        return continuation


class MBPP(BaseEval):
    mbpp_evaluator = evaluator.MBPPEvaluator()
    def get_task_names(self):
        return ["mbpp"]
    
    def get_answer(self, data):
        return data['test_list']
    
    def set_fewshot_dataset(self, dataset):
        self.few_shot_data = json.load(open("few_shot.json", encoding='utf-8'))[self.get_task_names()[0]]['ocp']

    def build_an_example_prompt(self, data, with_answer=True, flag=0):
        text = data['text']
        test_case = "\n".join(data['test_list'])
        prompt_in = f"You are an expert Python programmer, and here is your task: {text} Your code should pass these tests:\n\n {test_case} \n\n[BEGIN]\n"

        if with_answer:
            code = data["code"]
            prompt_out = f" '{code}' \n[DONE] \n\n "
            return prompt_in + prompt_out
        else:
            return prompt_in

    def build_prompt(self, task_name, data, dataset, shot):
        prompt = []
        for one_shot in self.few_shot_data:
            prompt.append(self.build_an_example_prompt(one_shot, with_answer=True))
        prompt.append(self.build_an_example_prompt(data, with_answer=False))
        prompt = "\n".join(prompt)
        return prompt

    def loaddataset(self, task_name, *args):
        # def processing_test(example):
        #     example['test_case'] = example['test_list']
        #     example['test_list'] = '\n'.join(example['test_list'])
        #     example['test_list_2'] = example['test_list']
        #     return example
        file_path = os.path.join(self.data_path, "mbpp.jsonl")
        train = datasets.load_dataset('json', data_files=file_path, split='train[:10]')
        test = datasets.load_dataset('json', data_files=file_path, split='train[10:510]')
        return DatasetDict({'train': train, 'test': test})

    def is_correct(self, prediction, test_case):
        result = self.mbpp_evaluator.score(prediction, test_case)
        if result == "pass":
            return 1
        else:
            return 0
        
    def parser_generation(self, continuation):
        continuation = continuation[0]
        return continuation
    

class HumanEval(BaseEval):
    humaneval_evaluator = evaluator.HumanEvaluator()

    def get_task_names(self):
        return ["humaneval"]

    def get_answer(self, data):
        return data['canonical_solution']
    
    def set_fewshot_dataset(self, dataset):
        self.few_shot_data = None

    def build_an_example_prompt(self, data, with_answer=True, flag=0):
        text = data['prompt']
        return text

    def build_prompt(self, task_name, data, dataset, shot):
        # prompt = "Complete the following python code:\n"
        prompt = ""
        prompt += self.build_an_example_prompt(data)
        return prompt

    def loaddataset(self, task_name):
        with open(os.path.join(self.data_path, 'test.jsonl'), encoding='utf-8') as f:
            data = [json.loads(x) for x in f.readlines()]
        dataset = datasets.DatasetDict({
            'test': datasets.Dataset.from_list(data),
        })
        return dataset
    
    def parser_generation(self, continuation):
        continuation = continuation[0]
        if 'phi-1d5' == self.model_id:
            continuation = continuation.split('\n\n\n')[0]
            # 观察发现phi-1d5最后一行的缩进多一个空格
            segs = continuation.split('\n')
            if segs[-1][: 5] == 5 * ' ':
                segs[-1] = segs[-1][1:]
            return '\n'.join(segs)
        return continuation

    def is_correct(self, pred, answer):
        return None

    def calc_acc(self):
        # humaneval_pass@1
        score = self.humaneval_evaluator.score(list(self.ttask_id2predict_label['humaneval'].values()))
        all_record = {
            "start_time": self.start_time,
            'end_time': get_time(),
            'md5_info': self.md5_info
        }
        record = dict()
        correct = []
        for k, v in self.ttask_id2details.items():
            correct += v
        instance_num = len(correct)
        # average_example = f'{score:.2f}'

        record['average_example'] = score
        record['instance_num'] = instance_num
        all_record['scores'] = record
        self.save_results({self.filename_pfx: all_record})


class HumanEval20240116(HumanEval):
    def build_prompt(self, task_name, data, dataset, shot):
        prompt = "Complete the following python code:\n"
        # prompt = ""
        prompt += self.build_an_example_prompt(data)
        return prompt

    def parser_generation(self, continuation):
        continuation = continuation[0]
        if 'gpt' in self.model_id:
            return self.humaneval_gpt_postprocess(continuation)
        else:
            return self.humaneval_postprocess_v2(continuation)
    
    def humaneval_postprocess_v2(self, text: str) -> str:
        """This is an advanced version of previous postprocess to handle more
        situations, better to use this one."""
        text = text.lstrip('\n')
        if '```' in text:
            blocks = re.findall(r'```(.*?)```', text, re.DOTALL)
            if len(blocks) == 0:
                text = text.split('```')[1]  # fall back to default strategy
            else:
                text = blocks[0]  # fetch the first code block
                if not text.startswith('\n'):  # in case starting with ```python
                    text = text[max(text.find('\n') + 1, 0):]
        if text.strip().startswith('from') or text.strip().startswith('import'):
            def_idx = text.find('def')
            if def_idx != -1:
                text = text[max(text.find('\n', def_idx) + 1, 0):]
        # remove empty lines
        text = '\n'.join([line for line in text.split('\n') if line != ''])
        text = text.lstrip('\n')
        if text.strip().startswith('def'):
            text = '\n'.join(text.split('\n')[1:])
        if not text.startswith('    '):
            if text.startswith(' '):
                text = '    ' + text.lstrip()
            else:
                text = '\n'.join(['    ' + line for line in text.split('\n')])
        text = text.split('\n')

        # If number of leading space reduces, we assume that the code block ends.
        min_leading_space = None
        end_index = None
        for index, line in enumerate(text):
            if line.strip() == '' or line.strip()[0] in ["'", '"', '#']:
                continue
            current_leading_space = len(line.rstrip()) - len(line.strip())
            if min_leading_space is None:
                min_leading_space = current_leading_space
            elif current_leading_space < min_leading_space:
                end_index = index
                break
        if end_index is not None:
            text = '\n'.join(text[:end_index])
        else:
            text = '\n'.join(text)
        return text


    def humaneval_gpt_postprocess(self, text: str) -> str:
        """Better answer postprocessor for better instruction-aligned models like
        GPT."""
        if '```' in text:
            blocks = re.findall(r'```(.*?)```', text, re.DOTALL)
            if len(blocks) == 0:
                text = text.split('```')[1]  # fall back to default strategy
            else:
                text = blocks[0]  # fetch the first code block
                if not text.startswith('\n'):  # in case starting with ```python
                    text = text[max(text.find('\n') + 1, 0):]
        if text.strip().startswith('from') or text.strip().startswith('import'):
            def_idx = text.find('def')
            if def_idx != -1:
                text = text[max(text.find('\n', def_idx) + 1, 0):]
        text = text.split('\n\n\n')[0]
        if text.strip().startswith('def'):
            text = '\n'.join(text.split('\n')[1:])
        if not text.startswith('    '):
            if text.startswith(' '):
                text = '    ' + text.lstrip()
            else:
                text = '\n'.join(['    ' + line for line in text.split('\n')])
        return text
            

class CoinFlip(BaseEval):
    def get_task_names(self):
        return ['coinflip']

    def loaddataset(self, task_name, *args):
        return datasets.load_dataset(self.data_path)

    def get_answer(self, data):
        # Yes or No
        return data['targets'][0].upper() + data['targets'][1:]

    def set_fewshot_dataset(self, dataset):
        self.few_shot_data = dataset['validation'].shuffle(seed=random_seed)

    def build_prompt(self, task_name, data, dataset, shot, shot_separator="\n\n"):
        prompt = ""
        for index in range(shot):
            example = self.few_shot_data[index]
            answer = example['targets'][0].upper() + example['targets'][1:]
            prompt += example['inputs'].strip() + f"\nA: {answer}{shot_separator}"

        prompt += data['inputs'].strip() + "\nA: "
        return prompt


class RuleTaker(BaseEval):
    def get_task_names(self):
        return ['ruletaker']

    def loaddataset(self, task_name, *args):
        return datasets.load_dataset(self.data_path)

    def get_answer(self, data):
        # Yes or No
        return {"entailment": "Yes", "not entailment": "No"}[data['label']]

    def set_fewshot_dataset(self, dataset):
        self.few_shot_data = dataset['validation'].shuffle(seed=random_seed)

    def build_prompt(self, task_name, data, dataset, shot, shot_separator="\n\n"):
        prompt = ""
        for index in range(shot):
            example = self.few_shot_data[index]
            prompt += f"Here are some facts:\n{example['context']}\n"
            question = example['question'].strip(".").strip()
            question = question[0].lower() + question[1:]
            answer = {"entailment": "Yes", "not entailment": "No"}[example['label']]
            prompt += f"Based on the above facts, can we infer the fact that {question}?\nAnswer: {answer}{shot_separator}"

        prompt += f"Here are some facts:\n{data['context']}\n"
        question = data['question'].strip(".").strip()
        question = question[0].lower() + question[1:]
        prompt += f"Based on the above facts, can we infer the fact that {question}?\nAnswer: "
        return prompt


class ProofWriter(BaseEval):
    def get_task_names(self):
        return ['proofwriter']

    def loaddataset(self, task_name, *args):
        return datasets.load_dataset(self.data_path)

    def get_answer(self, data):
        # Yes or No
        return data['answer']

    def set_fewshot_dataset(self, dataset):
        self.few_shot_data = dataset['validation'].shuffle(seed=random_seed)

    def build_prompt(self, task_name, data, dataset, shot, shot_separator="\n\n"):
        prompt = ""
        for index in range(shot):
            example = self.few_shot_data[index]
            prompt += f"Here are some facts:\n"
            for fact in example['facts']:
                prompt += fact + "\n"
            prompt += f"Based on the above facts, can we infer the fact that {example['question']}?\n"
            prompt += f"Answer: {example['answer']}{shot_separator}"

        prompt += f"Here are some facts:\n"
        for fact in data['facts']:
            prompt += fact + "\n"
        prompt += f"Based on the above facts, can we infer the fact that {data['question']}?\n"
        prompt += f"Answer: "
        return prompt


class CLUTRR(BaseEval):
    def get_task_names(self):
        return ['clutrr']

    def loaddataset(self, task_name, *args):
        return datasets.load_dataset(self.data_path)

    def get_answer(self, data):
        # Yes or No
        return data['answer']

    def set_fewshot_dataset(self, dataset):
        self.few_shot_data = dataset['validation'].shuffle(seed=random_seed)

    def build_prompt(self, task_name, data, dataset, shot, shot_separator="\n\n"):
        idx_list = ["A", "B", "C", "D"]
        prompt = ""
        for index in range(shot):
            example = self.few_shot_data[index]
            question = f"Based on the above, how is {example['persons'][0]} related to {example['persons'][1]}?"
            prompt += f"{example['context']}\n{question}\n"
            for idx, choice in zip(idx_list, example['choices']):
                prompt += f"{idx}. {choice}\n"
            prompt += f"Answer: {example['answer']}{shot_separator}"

        question = f"Based on the above, how is {data['persons'][0]} related to {data['persons'][1]}?"
        prompt += f"{data['context']}\n{question}\n"
        for idx, choice in zip(idx_list, data['choices']):
            prompt += f"{idx}. {choice}\n"
        prompt += f"Answer: "
        return prompt


class LongEval:
    def __init__(self, model_id, model, tokenizer, output_dir):
        import argparse
        from longeval import eval
        myparser = argparse.ArgumentParser()
        myparser.add_argument("--model-name-or-path", type=str, default='', help="model path")
        myparser.add_argument("--task", type=str, default='lines', help="Which evaluation task to use. currently support [topics, lines]")
        myparser.add_argument("--num_gpus", type=int, default=1, help="number of gpus to use")
        myparser.add_argument("--max_gpu_memory", type=int, default=40, help="max per gpu memory in GiB. A100 is 40 or 80.")
        myparser.add_argument("--longchat_flash_attn", action='store_true', help="Only apply to longchat models. Whether to enable flash attention to save memory, but slower.")
        myparser.add_argument("--longchat_ratio", type=int, default=8, help="Only apply to longchat models. Use ratio=8 for 16K context length model. Only ratio=8 is supported now.")
        myparser.add_argument("--eval_shortest_only", action='store_true', default=0, help="Only eval the shortest case for illustration purpose")
        myparser.add_argument("--test_dir", type=str, default="evaluation", help="Directory of the testcases")
        myparser.add_argument("--framework", type=str, default=None, help="Framework for serving")
        myparser.add_argument('-f', default='')
        myparser.add_argument('-c', type=str, default='task_config.json')
        myargs = myparser.parse_args()


        myargs.task = 'lines'
        myargs.test_dir = 'longeval/evaluation'
        output_dir = os.path.join(output_dir, 'LongEval', model_id)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        eval.longeval_test(model, tokenizer, output_dir, myargs)


class PPL2(BaseEval):
    def loaddataset(self, source):
        source2text = json.load(open(os.path.join(self.data_path, 'sampled_data.json')))
        self.text = source2text[source]

    def run(self):
        text = self.text
        encodings = self.tokenizer(text, return_tensors="pt")
        seq_len = encodings.input_ids.size(1)
        self.stride = self.other_args['stride']
        self.max_length = self.other_args['max_length']

        nlls = []
        prev_end_loc = 0
        for begin_loc in range(0, seq_len, self.stride):
            end_loc = min(begin_loc + self.max_length, seq_len)
            trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
            input_ids = encodings.input_ids[:, begin_loc:end_loc].to('cuda')
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = self.model(input_ids, labels=target_ids)

                # loss is calculated using CrossEntropyLoss which averages over valid labels
                # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
                # to the left by 1.
                neg_log_likelihood = outputs.loss

            nlls.append(neg_log_likelihood)

            prev_end_loc = end_loc
            if end_loc == seq_len:
                break
        all_record = dict()
        ppl = torch.exp(torch.stack(nlls).mean()).to(torch.float32).detach().cpu().numpy()
        record = {'ppl': f'{ppl:.1f}', 'stride': self.stride, 'max_length': self.max_length, 'text_length': len(text), 'token_length': seq_len}
        all_record['start_time'] = self.start_time
        all_record['end_time'] = get_time()
        all_record['scores'] = record
        self.save_results({self.filename_pfx: all_record})
        return {self.filename_pfx: record}


def general_postprocess(text: str) -> str:
    # Cut off the first newline, period, or comma
    truncated_text = re.split(r'[\n.,]', text, 1)[0]

    # Remove punctuation
    no_punctuation = re.sub(r'[^\w\s]', '', truncated_text)

    # Remove article
    no_articles = re.sub(r'\b(a|an|the)\b',
                         '',
                         no_punctuation,
                         flags=re.IGNORECASE)

    # Remove duplicated blank spaces
    cleaned_text = re.sub(r'\s+', ' ', no_articles).strip()

    return cleaned_text


def run_standard_task(config, task, model_id, model, tokenizer, md5_info, rank, args):
    output_dir = config['output_dir']
    task_config = config[task]
    shot = int(task_config.get('shot', 0))
    data_home = config['data_home'] if 'data_home' in config else ''
    data_path = os.path.join(data_home, task_config['data_path'])
    splits = task_config.get('split', 'test').split(',')
    
    for split in splits:
        generation_config = copy.deepcopy(config['generation_config'])
        task_settings = utils2.build_task_setting(model_id, task, split, shot, config)
        # 通用
        if task == 'ceval':
            myjob = CEval(task_settings, generation_config, config, data_path, output_dir, candidate_labels=['A', 'B', 'C', 'D'])
        elif task == 'mmlu':
            myjob = MMLU(task_settings, generation_config, config, data_path, output_dir, candidate_labels=['A', 'B', 'C', 'D'])
        elif task == 'cmmlu':
            myjob = CMMLU(task_settings, generation_config, config, data_path, output_dir, candidate_labels=['A', 'B', 'C', 'D'])
        elif task == 'gaokao':
            myjob = Gaokao(task_settings, generation_config, config, data_path, output_dir, candidate_labels=['A', 'B', 'C', 'D'])
        elif task == 'agieval':
            myjob = AGIEval(task_settings, generation_config, config, data_path, output_dir, candidate_labels=['A', 'B', 'C', 'D'])
        elif task == 'bcevalzh':
            myjob = BCEvalZh(task_settings, generation_config, config, data_path, output_dir, candidate_labels=['A', 'B', 'C', 'D'])
        # 法律金融
        elif task == 'jecqa':
            myjob = JecQA(task_settings, generation_config, config, data_path, output_dir, candidate_labels=['A', 'B', 'C', 'D'])
        elif task == 'financelq':
            myjob = FinancelQ(task_settings, generation_config, config, data_path, output_dir, candidate_labels=['A', 'B', 'C', 'D'])
        # 医疗
        elif task == 'medqa':
            myjob = MedQA(task_settings, generation_config, config, data_path, output_dir, candidate_labels=['A', 'B', 'C', 'D', 'E'])
        elif task == 'medmcqa':
            myjob = MedMCQA(task_settings, generation_config, config, data_path, output_dir, candidate_labels=['A', 'B', 'C', 'D'])
        elif task == 'medexam':
            myjob = MedExam(task_settings, generation_config, config, data_path, output_dir, candidate_labels=['A', 'B', 'C', 'D', 'E'])
        elif task == 'pubmedqa':
            myjob = PubMedQA(task_settings, generation_config, config, data_path, output_dir, candidate_labels=['A', 'B', 'C'])
        elif task == 'pubmedqa_v2':
            myjob = PubMedQA_V2(task_settings, generation_config, config, data_path, output_dir, candidate_labels=['A', 'B', 'C'])
        elif task == 'cmexam':
            myjob = CMExam(task_settings, generation_config, config, data_path, output_dir, candidate_labels=['A', 'B', 'C', 'D', 'E'])
        # 其它
        elif task == 'linguistic':
            myjob = Linguistic(task_settings, generation_config, config, data_path, output_dir, candidate_labels=['A', 'B', 'C', 'D'])
        elif task == 'quaternion':
            task_settings['task_action'] = utils2.ACTION_GENERATION
            myjob = QuaternionGen(task_settings, generation_config, config, data_path, output_dir)
        elif task == 'quaternionmixop':
            task_settings['task_action'] = utils2.ACTION_GENERATION
            myjob = QuaternionGenMixOp(task_settings, generation_config, config, data_path, output_dir)
        elif task == 'quaternionmc':
            myjob = QuaternionMC(task_settings, generation_config, config, data_path, output_dir, candidate_labels=['A', 'B', 'C', 'D'])
        elif task == 'selfbench':
            myjob = SelfBench(task_settings, generation_config, config, data_path, output_dir, candidate_labels=['A', 'B', 'C', 'D'])
        elif task == 'superclue0801':
            myjob = SuperClue0801(task_settings, generation_config, config, data_path, output_dir, candidate_labels=['A', 'B', 'C', 'D'])
        elif task == 'legalbench':
            myjob = LegalBench(task_settings, data_path, output_dir, output_dir)
        
        # 数学代码
        elif task == "gsm8k":
            myjob = GSM8K(task_settings, generation_config, config, data_path, output_dir)
        elif task == "math":
            myjob = MATH(task_settings, generation_config, config, data_path, output_dir)
        elif task == "humaneval":
            myjob = HumanEval(task_settings, generation_config, config, data_path, output_dir)
        elif task == "humaneval20240116":
            myjob = HumanEval20240116(task_settings, generation_config, config, data_path, output_dir)
        elif task == "mbpp":
            myjob = MBPP(task_settings, generation_config, config, data_path, output_dir)
        # 逻辑推理
        elif task == "clutrr":
            myjob = CLUTRR(task_settings, generation_config, config, data_path, output_dir, candidate_labels=['A', 'B', 'C', 'D'])
        elif task == "coinflip":
            myjob = CoinFlip(task_settings, generation_config, config, data_path, output_dir, candidate_labels=["Yes", "No"])
        elif task == "ruletaker":
            myjob = RuleTaker(task_settings, generation_config, config, data_path, output_dir, candidate_labels=["Yes", "No"])
        elif task == "proofwriter":
            myjob = ProofWriter(task_settings, generation_config, config, data_path, output_dir, candidate_labels=["Yes", "No"])
        elif task == 'itemorder':
            myjob = ItemOrder(task_settings, generation_config, config, data_path, output_dir, candidate_labels=['A', 'B', 'C', 'D'])
        elif task == 'itemcount':
            myjob = ItemCount(task_settings, generation_config, config, data_path, output_dir, candidate_labels=['A', 'B', 'C', 'D'])
        elif task == 'timeorder':
            myjob = TimeOrder(task_settings, generation_config, config, data_path, output_dir, candidate_labels=['A', 'B'])
        elif task == 'xingce':
            myjob = XingCe(task_settings, generation_config, config, data_path, output_dir, candidate_labels=['A', 'B', 'C', 'D'])
        elif task == 'xingcesl':
            myjob = XingCeSL(task_settings, generation_config, config, data_path, output_dir, candidate_labels=['A', 'B', 'C', 'D'])
        elif task == 'xingcetl':
            myjob = XingCeTL(task_settings, generation_config, config, data_path, output_dir, candidate_labels=['A', 'B', 'C', 'D'])
        # 新增
        elif task == 'bbh':
            task_settings['task_action'] = utils2.ACTION_GENERATION
            task_settings['shot'] = 3
            myjob = BBH(task_settings, generation_config, config, data_path, output_dir)
        elif task == 'nq':
            task_settings['task_action'] = utils2.ACTION_GENERATION
            myjob = NQ(task_settings, generation_config, config, data_path, output_dir)
        elif task == 'piqa':
            task_settings['task_action'] = utils2.ACTION_PPL
            myjob = PIQA(task_settings, generation_config, config, data_path, output_dir, candidate_labels=['A', 'B'])
        elif task == 'race':
            task_settings['task_action'] = utils2.ACTION_PPL
            myjob = RACE(task_settings, generation_config, config, data_path, output_dir, candidate_labels=['A', 'B', 'C', 'D'])
        elif task == 'arc':
            task_settings['task_action'] = utils2.ACTION_PPL
            myjob = ARC(task_settings, generation_config, config, data_path, output_dir, candidate_labels=['A', 'B', 'C', 'D'])
        elif task == 'hellaswag':
            task_settings['task_action'] = utils2.ACTION_PPL
            myjob = Hellaswag(task_settings, generation_config, config, data_path, output_dir, candidate_labels=['A', 'B', 'C', 'D'])
        elif task == 'winogrande':
            task_settings['task_action'] = utils2.ACTION_PPL
            myjob = Winogrande(task_settings, generation_config, config, data_path, output_dir, candidate_labels=['A', 'B', 'C', 'D'])
        elif task == 'lambada':
            task_settings['task_action'] = utils2.ACTION_GENERATION
            myjob = Lambada(task_settings, generation_config, config, data_path, output_dir)
        elif task == 'roleeval':
            myjob = RoleEval(task_settings, generation_config, config, data_path, output_dir, candidate_labels=['A', 'B', 'C', 'D'])
        elif task == 'cruxevalinput':
            task_settings['task_action'] = utils2.ACTION_GENERATION
            myjob = CRUXEvalInput(task_settings, generation_config, config, data_path, output_dir)
        elif task == 'cruxevaloutput':
            task_settings['task_action'] = utils2.ACTION_GENERATION
            myjob = CRUXEvalOutput(task_settings, generation_config, config, data_path, output_dir)
        # 英译中，中译英
        elif task == 'mmlutrans':
            myjob = MMLUTrans(task_settings, generation_config, config, data_path, output_dir, candidate_labels=['A', 'B', 'C', 'D'])
        elif task == 'cmmlutrans':
            myjob = CMMLUTrans(task_settings, generation_config, config, data_path, output_dir, candidate_labels=['A', 'B', 'C', 'D'])
        elif task == 'medqatrans':
            myjob = MedQATrans(task_settings, generation_config, config, data_path, output_dir, candidate_labels=['A', 'B', 'C', 'D', 'E'])
        else:
            continue
        myjob.set_model_and_tokenizer(model, tokenizer, md5_info)
        if rank == 0:
            print("generation_config: ", myjob.generation_config)
            print(f'StartTime {get_time()}\t{myjob.filename_pfx}')
        myjob.run(rank)
        if rank == 0:
            myjob.calc_acc()
            print(f'EndTime {get_time()}{myjob.filename_pfx}')
        del myjob


def run_new():
    rank = int(os.getenv("RANK", "0"))
    device = torch.device("cuda:{}".format(rank))

    args = utils2.get_args()
    config_filename = args.c
    config = utils2.get_config(config_filename)
    model_id_list = utils2.prepare_model_id(config['model_id'])
    task_list = utils2.prepare_task_list(config['tasks'])
    source2text = None
    data_home = config['data_home'] if 'data_home' in config else ''
    dtype = config.get('dtype', 'fp16')

    # model_id and model_path
    model_id2model_path = dict()
    for model_id in model_id_list:
        if config.get('model_home', '').strip() != "":
            model_path = os.path.join(config['model_home'], model_id)
        else:
            model_path = model_id
        model_id = model_path.split('/')[-1]
        model_id2model_path[model_id] = model_path  # model_id is the folder name of the model, model_path is the model folder path

    # status
    config_filename = config_filename.split('/')[-1]
    if rank == 0:
        if not os.path.exists(config['output_dir']):
            os.makedirs(config['output_dir'])
    status_filename = os.path.join(config['output_dir'], 'status2_' + config_filename.split('.')[0] + '.json')
    if args.f:  # 强制执行所有任务
        status_filename = os.path.join(config['output_dir'], 'tmp_status_' + config_filename.split('.')[0] + '.json')
        if rank == 0:
            json.dump({}, open(status_filename, 'w', encoding='utf-8'))
    zxrstatus = utils2.ZXRStaus(status_filename, model_id_list, task_list)
    if rank == 0:
        zxrstatus.init_status()
        zxrstatus.show_status()
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    # loop model and it's task
    status = zxrstatus.load_status()
    model_id_and_task_list = list(filter(lambda x: len(x[1]) > 0, status[utils2.STATUS_JOB_WAITING].items()))  # [(m1, [t1, t2]), (m2, [t1, t2, t3])]
    while len(model_id_and_task_list) > 0:
        model_id, model_task_list = model_id_and_task_list[0]  # model_id_and_task_list在每个循环中都会更新，每次取其最前面的元素
        model_path = model_id2model_path[model_id]
        model, tokenizer = None, None

        if len(model_task_list) > 0 \
            and not utils2.is_api(model_id) \
            and os.path.exists(model_path):
            model, tokenizer = utils2.load_model(model_path, dtype)
        md5_info = utils2.generate_model_md5(model_path)

        # 获取最新的状态
        status = zxrstatus.load_status()
        model_task_list = status[utils2.STATUS_JOB_WAITING].get(model_id, [])
        while len(model_task_list) > 0:
            task = model_task_list[0]  # model_task_list在每个循环中都会更新，每次取其最前面的元素
            try:
                if not torch.distributed.is_initialized() or int(os.getenv('RANK')) == 0:
                    zxrstatus.update_status(model_id, task, utils2.STATUS_JOB_RUNNING)
                if rank == 0:
                    zxrstatus.show_status()
                if 'pplmetric' == task.split('_')[0]:
                    ppl_metric_eval.run(model_id, task, config, model, tokenizer)
                elif 'ppl2' in task:
                    output_dir = config['output_dir']
                    task_config = config['ppl2']
                    data_path = os.path.join(data_home, task_config['data_path'])
                    source = task.split('_')[-1]
                    task_settings = OrderedDict({
                        'model_id': model_id,
                        'task': task,
                        'dtype': dtype,
                        })
                    other_args = task_config
                    myjob = PPL2(task_settings, data_path, output_dir)
                    myjob.loaddataset(source)
                    myjob.set_model_and_tokenizer(model, tokenizer, md5_info)
                    myjob.run()

                elif 'longbench' in task:
                    task_config = config['longbench']
                    data_path = os.path.join(data_home, task_config['data_path'])
                    max_length = task_config['max_length']
                    myjob = LongBench(model_id, data_path, config['output_dir'], max_length)
                    myjob.set_model_and_tokenizer(model, tokenizer)
                    myjob.run()
                elif 'words2sentencegenerate' in task:
                    model_config = config[model_id]
                    output_dir = config['output_dir']
                    task_config = config['words2sentencegenerate']
                    data_filename = os.path.join(data_home, task_config['data_path'])
                    shot = task_config['shot']
                    myjob = Words2SentenceGenerate(model_id, model, tokenizer, model_config, data_filename, output_dir, shot)
                    myjob.run()
                elif 'longeval' in task:
                    output_dir = config['output_dir']
                    myjob = LongEval(model_id, model, tokenizer, output_dir)
                else:
                    run_standard_task(copy.deepcopy(config), task, model_id, model, tokenizer, md5_info, rank, args)
                if not torch.distributed.is_initialized() or int(os.getenv('RANK')) == 0:
                    zxrstatus.update_status(model_id, task, utils2.STATUS_JOB_SUCESS)
            except Exception as e:
                if not torch.distributed.is_initialized() or int(os.getenv('RANK')) == 0:
                    zxrstatus.update_status(model_id, task, utils2.STATUS_JOB_FAIL)
                print(f'ZXRException message: {e}')
                traceback.print_exc()
            # 重新获取最新的状态
            status = zxrstatus.load_status()
            model_task_list = status[utils2.STATUS_JOB_WAITING].get(model_id, [])
        try:
            model = model.cpu()
            del model
        except:
            pass
        torch.cuda.empty_cache()

        # 重新获取最新的状态
        status = zxrstatus.load_status()
        model_id_and_task_list = list(filter(lambda x: len(x[1]) > 0, status[utils2.STATUS_JOB_WAITING].items()))  # [(m1, [t1, t2]), (m2, [t1, t2, t3])]

    if rank == 0:
        zxrstatus.show_status()
    if rank == 0:
        if args.f:
            os.remove(status_filename)

    print('\n' + 20 * '-' + 'Job Done' + 20 * '-')


def run_old():
    rank = int(os.getenv("RANK", "0"))
    device = torch.device("cuda:{}".format(rank))

    args = utils2.get_args()
    config_filename = args.c
    config = utils2.get_config(config_filename)
    model_id_list = utils2.prepare_model_id(config['model_id'])
    task_list = utils2.prepare_task_list(config['tasks'])
    source2text = None

    config_filename = config_filename.split('/')[-1]
    os.makedirs(config['output_dir'], exist_ok=True)
    # json_filename = os.path.join(config['output_dir'], 'status_' + config_filename.split('.')[0] + '.json')
    json_filename = os.path.join(config['output_dir'], "status_task_global.json")
    # if args.f:  # 强制执行所有任务
    #     postfix = f"_{int(time.time() % 10 * 1e9 // 1)}"
    #     json_filename = os.path.join(config['output_dir'], 'tmp_status_' + config_filename.split('.')[0] + postfix + '.json')
    #     if rank == 0:
    #         json.dump({}, open(json_filename, 'w', encoding='utf-8'))
    if rank == 0:
        print(f"status_file: {json_filename}")

    data_home = config['data_home'] if 'data_home' in config else ''
    # random.Random(time.localtime()).shuffle(model_id_list)

    dtype = config.get('dtype', 'fp16')

    for model_id in model_id_list:
        if config.get('model_home', '').strip() != "":
            model_path = os.path.join(config['model_home'], model_id)
        else:
            model_path = model_id
        if model_path not in utils2.OPENAI_APIS \
                and model_id not in utils2.ZXRExpMODEL \
                and not utils2.is_api(model_path) \
                and not os.path.exists(model_path):
            if rank == 0:
                print(f'\n\nModelNotExist: {model_id}\n\n')
            continue

        is_load_model = False
        model, tokenizer = None, None
        model_id = model_id.split('/')[-1]

        for task in task_list:
            if rank == 0:
                status = utils2.get_status(json_filename, model_id_list, task_list, model_id)
                if task in status.get(model_id, []) and not args.f:
                    # if task in status.get(model_id, []):
                    continue
                utils2.get_status(json_filename, model_id_list, task_list, model_id, task, is_show=True)
            if is_load_model == False:
                md5_info = utils2.generate_model_md5(model_path, skip=True)
                model, tokenizer = utils2.load_model(model_path, dtype, rank)
                is_load_model = True
            try:
                if 'pplmetric' == task.split('_')[0]:
                    ppl_metric_eval.run(model_id, task, config, model, tokenizer)
                elif 'ppl2' in task:
                    output_dir = config['output_dir']
                    task_config = config['ppl2']
                    data_path = os.path.join(data_home, task_config['data_path'])
                    source = task.split('_')[-1]
                    task_settings = OrderedDict({
                        'model_id': model_id,
                        'task': task,
                        'dtype': dtype,
                    })
                    other_args = task_config
                    myjob = PPL2(task_settings, data_path, output_dir)
                    myjob.loaddataset(source)
                    myjob.set_model_and_tokenizer(model, tokenizer, md5_info)
                    myjob.run()

                elif 'longbench' in task:
                    task_config = config['longbench']
                    data_path = os.path.join(data_home, task_config['data_path'])
                    max_length = task_config['max_length']
                    myjob = LongBench(model_id, data_path, config['output_dir'], max_length)
                    myjob.set_model_and_tokenizer(model, tokenizer)
                    myjob.run()
                elif 'words2sentencegenerate' in task:
                    model_config = config[model_id]
                    output_dir = config['output_dir']
                    task_config = config['words2sentencegenerate']
                    data_filename = os.path.join(data_home, task_config['data_path'])
                    shot = task_config['shot']
                    myjob = Words2SentenceGenerate(model_id, model, tokenizer, model_config, data_filename, output_dir, shot)
                    myjob.run()
                elif 'longeval' in task:
                    output_dir = config['output_dir']
                    myjob = LongEval(model_id, model, tokenizer, output_dir)
                else:
                    run_standard_task(copy.deepcopy(config), task, model_id, model, tokenizer, md5_info, rank, args)
            except Exception as e:
                print(f'ZXRException message: {e}')
                traceback.print_exc()

        if model is not None:
            model = model.cpu()
            # try:
            #     model = model.cpu()
            # except Exception as e:
            #     print(e)

            del model
        torch.cuda.empty_cache()
    if rank == 0:
        # if args.f:
        #     os.remove(json_filename)
        print('\n' + 20 * '-' + 'Job Done' + 20 * '-')


if __name__ == '__main__':
    print(f'pid: {os.getpid()}')
    # run_new()
    run_old()
