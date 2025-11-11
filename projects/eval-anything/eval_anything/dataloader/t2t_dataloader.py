# Copyright 2025 PKU-Alignment Team. All Rights Reserved.
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
t2t dataloader基类
输入：
    - 数据集路径
    - split
    - size
    - 模态
    - 预处理方式（是否pre-tokenize）
    - 模型路径（如果需要pre-tokenize）
    - shuffle
    - num_workers
    - chat_template
    - ...
输出：
    - InferenceInput类
"""

import os
from collections import namedtuple
from typing import Dict, List

from datasets import load_dataset

from eval_anything.dataloader.base_dataloader import BaseDataLoader
from eval_anything.utils.data_type import InferenceInput
from eval_anything.utils.register import DataloaderRegistry, PromptBuilderRegistry
from eval_anything.utils.utils import get_project_root


TASK_TYPE_MAP = {
    'Dialogue': 'build_dialogue_prompt',
    'DialogueChinese': 'build_dialogue_chinese_prompt',
    'MultiChoice': 'build_multi_choice_prompt',
    'MultiChoiceChinese': 'build_multi_choice_prompt_chinese',
    'CodeGeneration': 'build_codes_generation_prompt',
}


@DataloaderRegistry.register('T2TDataloader')
class T2TDataLoader(BaseDataLoader):

    def __init__(self, eval_cfgs, bench_cfgs, logger):
        super().__init__(eval_cfgs, bench_cfgs, logger)
        self.TASK_TYPE_MAP = TASK_TYPE_MAP

    def load_dataset(self, task_list: list[str]) -> Dict[str, List[InferenceInput]]:
        prompts = {}
        if task_list == []:
            task_list = list(self.bench_cfgs.dataset.default_task_list)
        for task in self.task_info:
            if task.name not in task_list:
                continue
            if task.data_files:
                dataset = load_dataset(self.data_dir, data_files=task.data_files, split='train')
            else:
                dataset = load_dataset(
                    self.data_dir, task.name, split=self.bench_cfgs.dataset.split
                )
            self.few_shot_data[task.name] = (
                self.set_fewshot_dataset(task.name) if self.num_shot != 0 else None
            )

            # Build promptsxi
            prompt_builder = getattr(self, self.TASK_TYPE_MAP[task.type])
            prompts[task.name] = prompt_builder(task, dataset)

            # Save metadata if needed
            if task.has_metadata:
                for i, item in enumerate(prompts[task.name]):
                    item.metadata = dataset[i]

        return prompts

    def set_fewshot_dataset(self, task: str):
        if self.enable_cot:
            cot_fewshot_data_split = (
                self.bench_cfgs.dataset.cot_fewshot_data_split
                if self.bench_cfgs.dataset.cot_fewshot_data_split
                else 'train'
            )
            try:
                data_path = os.path.join(
                    get_project_root(),
                    'eval_anything',
                    self.bench_cfgs.dataset.cot_fewshot_data_path,
                )
                if os.path.exists(data_path):
                    few_shot_data = load_dataset(
                        data_path,
                        data_files=self.bench_cfgs.dataset.cot_fewshot_data_file,
                        split=cot_fewshot_data_split,
                    )
                    return few_shot_data
                else:
                    few_shot_data = load_dataset(
                        self.bench_cfgs.dataset.cot_fewshot_data_path,
                        name=task,
                        data_files=self.bench_cfgs.dataset.cot_fewshot_data_file,
                        split=cot_fewshot_data_split,
                    )
                    return few_shot_data
            except:
                self.logger.log(
                    'error',
                    f'Chain of thought fewshot is not supported for task {self.bench_cfgs.dataset.name}: {task}',
                )
                raise
        else:
            fewshot_data_split = (
                self.bench_cfgs.dataset.fewshot_data_split
                if self.bench_cfgs.dataset.fewshot_data_split
                else 'train'
            )
            try:
                data_path = os.path.join(
                    get_project_root(), 'eval_anything', self.bench_cfgs.dataset.fewshot_data_path
                )
                if os.path.exists(data_path):
                    few_shot_data = load_dataset(
                        data_path,
                        data_files=self.bench_cfgs.dataset.fewshot_data_file,
                        name=task,
                        split=fewshot_data_split,
                    )
                    return few_shot_data
                else:
                    few_shot_data = load_dataset(
                        self.bench_cfgs.dataset.fewshot_data_path,
                        name=task,
                        data_files=self.bench_cfgs.dataset.fewshot_data_file,
                        split=fewshot_data_split,
                    )
                    return few_shot_data
            except:
                self.logger.log(
                    'error',
                    f'Fewshot is not supported for task {self.bench_cfgs.dataset.name}: {task}',
                )
                raise

    def build_conversation_from_prompt(self, prompt: str):
        """
        Build a default conversation from a text prompt.
        """
        conversation = [{'role': 'user', 'content': prompt}]
        return conversation

    def build_multi_choice_prompt(self, task: namedtuple, data: list[dict]):
        few_shot_examples = self.few_shot_data[task.name][: self.num_shot] if self.num_shot else []

        prompt_builder = PromptBuilderRegistry.get_prompt_builder('MultiChoice')(
            candidate_labels=task.candidate_labels,
            few_shot_examples=few_shot_examples,
            cot=self.enable_cot,
        )
        prompts = []
        question_key = task.question_key
        answer_key = task.answer_key
        ground_truth_key = task.ground_truth_key

        for item in data:
            prompt = prompt_builder.build_prompt(
                item[question_key], item, question_key, answer_key, ground_truth_key
            )
            conversation = self.build_conversation_from_prompt(prompt)
            prompts.append(
                InferenceInput(
                    task=task.name, conversation=conversation, ref_answer=item[ground_truth_key]
                )
            )

        return prompts

    def build_multi_choice_prompt_chinese(self, task: namedtuple, data: list[dict]):
        few_shot_examples = self.few_shot_data[task.name][: self.num_shot] if self.num_shot else []
        prompt_builder = PromptBuilderRegistry.get_prompt_builder('MultiChoiceChinese')(
            candidate_labels=task.candidate_labels,
            few_shot_examples=few_shot_examples,
            cot=self.enable_cot,
        )
        prompts = []
        question_key = task.question_key
        answer_key = task.answer_key
        ground_truth_key = task.ground_truth_key

        for item in data:
            prompt = prompt_builder.build_prompt(item[question_key], item[answer_key])
            conversation = self.build_conversation_from_prompt(prompt)
            prompts.append(
                InferenceInput(
                    task=task.name, conversation=conversation, ref_answer=item[ground_truth_key]
                )
            )

        return prompts

    def build_dialogue_prompt(self, task: namedtuple, data: list[dict]):
        few_shot_examples = self.few_shot_data[task.name][: self.num_shot] if self.num_shot else []
        prompt_builder = PromptBuilderRegistry.get_prompt_builder('Dialogue')(
            few_shot_examples=few_shot_examples, cot=self.enable_cot
        )
        question_key = task.question_key
        ground_truth_key = task.ground_truth_key
        prompts = []
        for item in data:
            prompt = prompt_builder.build_prompt(item[question_key])
            conversation = self.build_conversation_from_prompt(prompt)
            prompts.append(
                InferenceInput(
                    task=task.name, conversation=conversation, ref_answer=item[ground_truth_key]
                )
            )

        return prompts

    def build_dialogue_chinese_prompt(self, task: namedtuple, data: list[dict]):
        few_shot_examples = self.few_shot_data[task.name][: self.num_shot] if self.num_shot else []
        prompt_builder = PromptBuilderRegistry.get_prompt_builder('DialogueChinese')(
            few_shot_examples=few_shot_examples, cot=self.enable_cot
        )
        question_key = task.question_key
        ground_truth_key = task.ground_truth_key
        prompts = []
        for item in data:
            prompt = prompt_builder.build_prompt(item[question_key])
            conversation = self.build_conversation_from_prompt(prompt)
            prompts.append(
                InferenceInput(
                    task=task.name, conversation=conversation, ref_answer=item[ground_truth_key]
                )
            )

        return prompts

    def build_codes_generation_prompt(self, task: namedtuple, data: list[dict]):
        few_shot_examples = self.few_shot_data[task.name][: self.num_shot] if self.num_shot else []
        prompt_builder = PromptBuilderRegistry.get_prompt_builder('CodesGeneration')(
            few_shot_examples=few_shot_examples, cot=self.enable_cot, language=task.language
        )
        prompts = []
        question_key = task.question_key
        ground_truth_key = task.ground_truth_key
        for item in data:
            prompt = prompt_builder.build_prompt(item[question_key], item[ground_truth_key])
            conversation = self.build_conversation_from_prompt(prompt)
            prompts.append(
                InferenceInput(
                    task=task.name, conversation=conversation, ref_answer=item[ground_truth_key]
                )
            )

        return prompts
