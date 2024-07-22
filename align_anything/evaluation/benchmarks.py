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
import os
import re
from typing import Any, Dict, List

import deepspeed
import pandas as pd
import torch
from PIL import Image

from align_anything.evaluation.base import BaseEvaluator
from align_anything.evaluation.categories import GaokaoCategories, MMECategories, MMLUCategories
from align_anything.evaluation.evaluator import GSM8KEvaluator
from align_anything.evaluation.evaluator_registry import get_template_class, register_evaluator
from align_anything.evaluation.mt_bench import MTBench
from align_anything.utils.multi_process import get_current_device, is_main_process
from align_anything.utils.tools import (
    custom_cfgs_to_dict,
    dict_to_namedtuple,
    read_cfgs,
    seed_everything,
    update_dict,
)
from datasets import Dataset, DatasetDict, load_dataset


@register_evaluator('mmlu')
class MMLU(BaseEvaluator):
    def get_task_names(self) -> List[str]:
        return list(sorted(MMLUCategories.keys()))

    def get_answer(self, data):
        return chr(65 + data.get('answer', 'N'))

    def set_fewshot_dataset(self, dataset: DatasetDict) -> None:
        self.few_shot_data = dataset['dev']

    def build_example_prompt(self, data, with_answer=True):
        choices = '\n'.join(
            [f'{label}: {data["choices"][ord(label) - 65]}' for label in self.candidate_labels]
        )
        answer = f'Answer: {self.get_answer(data)}' if with_answer else 'Answer: '
        return f"{data['question']}\n{choices}\n{answer}"

    def build_prompt(self, data):
        prompt = f'The following are multiple choice questions (with answers).\n\n'
        few_shot_examples = self.few_shot_data[: self.num_shot] if self.num_shot else []
        if len(few_shot_examples) == 0:
            return prompt + self.build_example_prompt(data, False)
        else:
            examples = [
                self.build_example_prompt(
                    {key: value[i] for key, value in few_shot_examples.items()}, True
                )
                for i in range(len(few_shot_examples['question']))
            ]
            examples.append(self.build_example_prompt(data, False))
            return prompt + '\n\n'.join(examples)


@register_evaluator('gaokao')
class GaokaoSingleChoice(BaseEvaluator):
    def get_answer(self, data) -> str:
        return data['answer'] if 'answer' in data else 'NoAnswer'

    def set_fewshot_dataset(self, dataset: DatasetDict) -> None:
        self.few_shot_data = dataset['train']

    def load_dataset(self, task_name):
        return load_dataset(
            'json',
            data_files={
                split: os.path.join(self.task_dir, split, GaokaoCategories[task_name])
                for split in ('train', 'dev')
            },
        )

    def get_task_names(self):
        return list(sorted(GaokaoCategories.keys()))

    def build_example_prompt(self, data, with_answer=True):
        question = data['question']
        choices = '\n'.join([f'{label}: {data[label]}' for label in self.candidate_labels])
        answer = f"答案：{data['answer']}" if with_answer else '答案：'
        return f'{question}\n{choices}\n{answer}'

    def build_prompt(self, data):
        few_shot_examples = self.few_shot_data[: self.num_shot] if self.num_shot else []
        if len(few_shot_examples) == 0:
            return self.build_example_prompt(data, False)
        else:
            examples = [
                self.build_example_prompt(
                    {key: value[i] for key, value in few_shot_examples.items()}, True
                )
                for i in range(len(few_shot_examples['question']))
            ]
            examples.append(self.build_example_prompt(data, False))
            return '\n'.join(examples)


@register_evaluator('gsm8k')
class GSM8K(BaseEvaluator):
    gsm8k_evalutor = GSM8KEvaluator()

    def get_task_names(self):
        return ['main']

    def get_answer(self, data):
        return self.gsm8k_evalutor._decimal_separator.sub('', data['answer'])

    def build_example_prompt(self, data, with_answer=True):
        question = data['question']
        prompt = f"Question: {question} Let's think step by step\nAnswer:\n"
        if with_answer:
            return prompt + self.gsm8k_evalutor._decimal_separator.sub('', data['answer'])
        else:
            return prompt

    def set_fewshot_dataset(self, dataset):
        self.few_shot_data = dataset['train']

    def build_prompt(self, data):
        prompt = ''
        few_shot_examples = self.few_shot_data[: self.num_shot] if self.num_shot else []
        examples = [
            self.build_example_prompt(
                {key: value[i] for key, value in few_shot_examples.items()}, True
            )
            for i in range(len(few_shot_examples['question']))
        ]
        examples.append(self.build_example_prompt(data, False))
        return prompt + '\n\n'.join(examples)

    def is_correct(self, prediction, reference):
        return self.gsm8k_evalutor.score(prediction, reference)

    def parser_response(self, response):
        response = response[0]
        return response.lstrip()


@register_evaluator('hellaswag')
class Hellaswag(BaseEvaluator):
    def get_task_names(self):
        return ['default']

    def get_answer(self, data):
        return 'ABCDEFG'[int(data['label'])]

    def set_fewshot_dataset(self, dataset):
        self.few_shot_data = None

    def build_prompt(self, data):
        assert self.num_shot == 0, 'Hellaswag does not support few-shot learning.'
        question = data['ctx']
        choices = data['endings']
        prompt = [question + ' ' + choice for choice in choices]
        return prompt

    def preproccess(self, data):
        prompts = self.build_prompt(data)
        inputs = [self.processor(prompt, return_tensors='pt').to(self.device) for prompt in prompts]
        answers = self.get_answer(data)

        return {
            'inputs': inputs,
            'answers': answers,
            'prompts': prompts,
        }


@register_evaluator('winogrande')
class Winogrande(BaseEvaluator):
    def get_task_names(self):
        task_names = [
            'winogrande',
        ]
        return task_names

    def load_dataset(self, task_name):
        print(task_name)
        filename = os.path.join(self.task_dir, 'dev.jsonl')
        with open(filename, encoding='utf-8') as f:
            data = [json.loads(x) for x in f.readlines()]
        for d in data:
            d['id'] = d['qID']
        dataset = DatasetDict(
            {
                'test': Dataset.from_list(data),
            }
        )
        return dataset

    def get_answer(self, data):
        return 'ABCDEFG'[int(data['answer']) - 1] if 'answer' in data else 'NoAnswer'

    def set_fewshot_dataset(self, dataset):
        self.few_shot_data = None

    def build_prompt(self, data):
        assert self.num_shot == 0, 'Winogrande does not support few-shot learning.'
        question = data['sentence']
        choices = [data[key] for key in ('option1', 'option2')]
        prompt = [question.replace('_', can) for can in choices]
        return prompt

    def preproccess(self, data):
        prompts = self.build_prompt(data)
        inputs = [self.processor(prompt, return_tensors='pt').to(self.device) for prompt in prompts]
        answers = self.get_answer(data)

        return {
            'inputs': inputs,
            'answers': answers,
            'prompts': prompts,
        }


@register_evaluator('mme')
class MME(BaseEvaluator):
    def get_task_names(self):
        return list(MMECategories.keys())

    def load_dataset(self, task_name: str) -> DatasetDict:
        data = []
        questions_dir = os.path.join(self.task_dir, task_name, 'questions_answers_YN')
        image_dir = os.path.join(self.task_dir, task_name, 'images')
        image_files = os.listdir(image_dir)
        for image_file in image_files:
            prefix = image_file.split('.')[0]
            qa_path = os.path.join(questions_dir, prefix + '.txt')
            image_path = os.path.join(image_dir, image_file)

            with open(qa_path) as f:
                text = f.readlines()
            for qa in text:
                question, answer = qa.strip().split('\t')
                data.append(
                    {
                        'question': question,
                        'answer': answer,
                        'image_path': image_path,
                        'id': prefix,
                    }
                )
        return DatasetDict({'val': Dataset.from_list(data)})

    def set_fewshot_dataset(self, dataset):
        self.few_shot_data = None

    def build_prompt(self, data: Dict[str, Any]) -> str:
        assert self.num_shot == 0, 'MME does not support few-shot learning.'
        return f"USER: <image>\n{data['question']}\nASSISTANT: "

    def parser_response(self, response):
        # TODO: batch processing
        response_clean = re.sub(r'[\s\n\t]+', '', response[0]).lower()

        if re.match(r'^yes$', response_clean):
            return 'yes'
        elif re.match(r'^no$', response_clean):
            return 'no'
        else:
            return 'unknown'

    def preproccess(self, data):
        image_path = data['image_path']
        raw_image = Image.open(image_path)
        prompt = self.build_prompt(data)

        inputs = self.processor(prompt, raw_image, return_tensors='pt').to(self.device)

        return {
            'inputs': inputs,
            'answers': data['answer'].lower(),
            'prompts': prompt,
            'id': data['id'],
        }

    def eval_instance(self, instance: Dict[str, Any]) -> Dict[str, Any]:
        details, correction = [], []
        preds, infos = self.predict(instance)

        for id, prompt, answer, pred, info in zip(
            instance['id'], instance['prompts'], instance['answers'], preds, infos
        ):
            is_correct = self.is_correct(pred, answer)

            detail = {
                'id': id,
                'prompt': prompt,
                'pred': pred,
                'answer': answer,
                'is_correct': is_correct,
                **info,
            }

            details.append(detail)
            correction.append(is_correct)
        return details, correction

    def calculate_results(self) -> None:
        acc = {'average': -1}
        total_correct = 0
        total_length = 0
        for task, correction in self.task2correction.items():
            acc[task] = sum(correction) / len(correction) if len(correction) > 0 else -1
            total_correct += sum(correction)
            total_length += len(correction)
        acc['average'] = total_correct / total_length

        acc_plus = {'average': -1}
        total_correct_plus = 0
        total_length_plus = 0
        for task, details in self.task2details.items():
            image_items = {}
            for detail in details:
                if detail['id'] not in image_items:
                    image_items[detail['id']] = {'is_correct': None, 'results': []}
                image_items[detail['id']]['results'].append(detail['is_correct'])
            for id, items in image_items.items():
                image_items[id]['is_correct'] = all(items['results'])
            acc_plus[task] = sum([item['is_correct'] for item in image_items.values()]) / len(
                image_items
            )
            total_correct_plus += sum([item['is_correct'] for item in image_items.values()])
            total_length_plus += len(image_items)
        acc_plus['average'] = total_correct_plus / total_length_plus

        score = {}
        keys = ['average'] + list(self.task2correction.keys())
        for key in keys:
            score[key] = (acc[key] + acc_plus[key]) * 100

        result = {
            'score': score,
            'accuracy': acc,
            'accuracy_plus': acc_plus,
        }

        with open(os.path.join(self.output_dir, self.results_filename), 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=4)


def main():
    # setup distribution
    deepspeed.init_distributed()
    current_device = get_current_device()
    torch.cuda.set_device(current_device)

    # get custom configs from command line
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    _, unparsed_args = parser.parse_known_args()

    # read default configs from the yaml file
    task = unparsed_args[-1]
    dict_configs, ds_configs = read_cfgs(mode='evaluation', task=task)

    keys = [k[2:] for k in unparsed_args[1::2]]
    values = list(unparsed_args[2::2])
    unparsed_args = dict(zip(keys, values))
    for k, v in unparsed_args.items():
        dict_configs = update_dict(dict_configs, custom_cfgs_to_dict(k, v))

    configs = dict_to_namedtuple(dict_configs)

    evaluator = get_template_class(task, configs.default, ds_configs)
    evaluator.eval()


if __name__ == '__main__':
    main()
