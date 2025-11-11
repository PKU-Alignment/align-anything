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

from collections import namedtuple
from typing import Optional

from eval_anything.dataloader.t2t_dataloader import T2TDataLoader
from eval_anything.utils.data_type import InferenceInput
from eval_anything.utils.logger import EvalLogger
from eval_anything.utils.register import DataloaderRegistry, PromptBuilderRegistry


@DataloaderRegistry.register('TruthfulQA')
class TruthfulQADataloader(T2TDataLoader):
    def __init__(self, eval_cfgs: namedtuple, benchmark_cfgs: namedtuple, logger: EvalLogger):
        super().__init__(eval_cfgs, benchmark_cfgs, logger)
        self.TASK_TYPE_MAP = {
            'DialogueWithAnswer': 'build_dialogue_with_answer_prompt',
            'DialogueList': 'build_dialogue_list_prompt',
            'MultiChoiceAutoLabel': 'build_multi_choice_auto_label_prompt',
        }

    # for mc1_targets of truthfulqa multiple_choice subset currently
    def build_multi_choice_auto_label_prompt(self, task: namedtuple, data: list[dict]):
        few_shot_examples = self.few_shot_data[task.name][: self.num_shot] if self.num_shot else []

        prompt_builder = PromptBuilderRegistry.get_prompt_builder('MultiChoiceAutoLabel')(
            few_shot_examples=few_shot_examples, cot=self.enable_cot
        )
        prompts = []
        question_key = task.question_key
        answer_key = task.answer_key
        ground_truth_key = task.ground_truth_key
        for item in data:
            prompt = prompt_builder.build_prompt(item[question_key], item[answer_key]['choices'])
            conversation = self.build_conversation_from_prompt(prompt)
            prompts.append(
                InferenceInput(
                    task=task.name,
                    conversation=conversation,
                    ref_answer=item[ground_truth_key]['labels'],
                )
            )

        return prompts

    def build_dialogue_with_answer_prompt(self, task: namedtuple, data: list[dict]):
        few_shot_examples = self.few_shot_data[task.name][: self.num_shot] if self.num_shot else []
        prompt_builder = PromptBuilderRegistry.get_prompt_builder('Dialogue')(
            few_shot_examples=few_shot_examples, cot=self.enable_cot
        )
        question_key = task.question_key
        best_ground_truth_key = task.best_ground_truth_key
        ground_truth_key = task.ground_truth_key
        anti_ground_truth_key = task.anti_ground_truth_key

        prompts = []
        idx = 0
        for item in data:
            best_answer_index = item[ground_truth_key].index(item[best_ground_truth_key])
            for correct_answer in item[ground_truth_key]:
                prompt = prompt_builder.build_prompt(item[question_key], correct_answer)
                conversation = self.build_conversation_from_prompt(prompt)
                prompts.append(
                    InferenceInput(
                        task=task.name,
                        conversation=conversation,
                        ref_answer=best_answer_index,
                        metadata=idx,
                    )
                )
            idx += 1
            for incorrect_answer in item[anti_ground_truth_key]:
                prompt = prompt_builder.build_prompt(item[question_key], incorrect_answer)
                conversation = self.build_conversation_from_prompt(prompt)
                prompts.append(
                    InferenceInput(
                        task=task.name,
                        conversation=conversation,
                        ref_answer=best_answer_index,
                        metadata=idx,
                    )
                )
        return prompts

    # for bleurt metric of truthfulqa currently
    def build_dialogue_list_prompt(self, task: namedtuple, data: list[dict]):
        few_shot_examples = self.few_shot_data[task.name][: self.num_shot] if self.num_shot else []
        prompt_builder = PromptBuilderRegistry.get_prompt_builder('Dialogue')(
            few_shot_examples=few_shot_examples, cot=self.enable_cot
        )
        question_key = task.question_key
        ground_truth_key = task.ground_truth_key
        anti_ground_truth_key = task.anti_ground_truth_key

        for item in data:
            if 'I have no comment.' not in item[ground_truth_key]:
                item[ground_truth_key].append('I have no comment.')

        prompts = []
        for item in data:
            prompt = prompt_builder.build_prompt(item[question_key])
            conversation = self.build_conversation_from_prompt(prompt)
            prompts.append(
                InferenceInput(
                    task=task.name,
                    conversation=conversation,
                    ref_answer={
                        'correct_answers': item[ground_truth_key],
                        'incorrect_answers': item[anti_ground_truth_key],
                    },
                )
            )

        return prompts


# Prompt Builder Registry
@PromptBuilderRegistry.register('MultiChoiceAutoLabel')
class MultiChoiceAutoLabelPromptBuilder:
    def __init__(
        self,
        multi_choice_prompt: Optional[str] = None,
        cot_context: Optional[str] = None,
        few_shot_examples: Optional[list[str]] = None,
        cot: bool = False,
    ):
        self.multi_choice_prompt = (
            multi_choice_prompt
            if multi_choice_prompt
            else 'Now please answer the following multiple choice question.'
        )
        self.cot_context = cot_context if cot_context else "Let's think step by step."
        self.few_shot_examples = few_shot_examples
        self.enable_cot = cot

    def marge_QA(self, question: str, candidate_answers: list[str], ground_truth: str = '') -> str:
        candidate_labels = [chr(65 + i) for i in range(len(candidate_answers))]

        if ground_truth.isdigit():
            ground_truth = [candidate_labels[int(ground_truth)]]
        prompt = f'{question}\n'
        for label, answer in zip(candidate_labels, candidate_answers):
            prompt += f'({label}) {answer} '

        if ground_truth:
            answer = f'\nAnswer: ({ground_truth})'
        else:
            answer = ''

        return prompt + answer + '\n'

    def build_prompt(self, question: str, candidate_answers: list[str]) -> str:
        prompt = ''

        if self.few_shot_examples:
            prompt += 'The following are multiple choice questions with answers.\n'
            for q, c, a in zip(
                self.few_shot_examples['question'],
                self.few_shot_examples['choices'],
                self.few_shot_examples['answer'],
            ):
                prompt += self.marge_QA(q, c, str(a))

        prompt += f'{self.multi_choice_prompt}\n\n'

        prompt += self.marge_QA(question, candidate_answers)
        if self.enable_cot:
            prompt += f'\n{self.cot_context}'
        return prompt
