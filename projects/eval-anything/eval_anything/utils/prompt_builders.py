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

from typing import Optional, Union

from eval_anything.evaluate_tools.base_tools import PromptBuilder
from eval_anything.utils.register import PromptBuilderRegistry


@PromptBuilderRegistry.register('MultiChoice')
class MultiChoicePromptBuilder(PromptBuilder):
    def __init__(
        self,
        candidate_labels: list[str],
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
        self.candidate_labels = candidate_labels
        self.cot_context = cot_context if cot_context else "Let's think step by step."
        self.few_shot_examples = few_shot_examples
        self.enable_cot = cot

    def marge_QA(self, question: str, candidate_answers: list[str], ground_truth: str = '') -> str:
        # Handle the ground_truth label mapping
        if ground_truth.isdigit():
            ground_truth = [self.candidate_labels[int(ground_truth)]]
        prompt = f'{question}\n'
        for label, answer in zip(self.candidate_labels, candidate_answers):
            prompt += f'({label}) {answer} '

        if ground_truth:
            answer = f'\nAnswer: ({ground_truth})'
        else:
            answer = ''

        return prompt + answer + '\n'

    def build_prompt(
        self,
        question: str,
        data_item: dict,
        question_key: str = 'question',
        answer_key: Union[tuple, list, str] = 'choices',
        ground_truth_key: str = 'answer',
    ) -> str:
        prompt = ''

        if self.few_shot_examples:
            # Add few-shot examples
            prompt += 'The following are multiple choice questions with answers.\n'
            for q, c, a in zip(
                self.few_shot_examples[question_key],
                self.few_shot_examples[answer_key],
                self.few_shot_examples[ground_truth_key],
            ):
                prompt += self.marge_QA(q, c, str(a)) + '\n'

        prompt += f'{self.multi_choice_prompt}\n\n'

        if isinstance(answer_key, tuple):
            answer_key = answer_key._asdict()
            value = data_item
            for key in answer_key.keys():
                value = value[key]
            candidate_answers = value
        elif isinstance(answer_key, list):
            candidate_answers = [data_item[answer_key_single] for answer_key_single in answer_key]
        else:
            candidate_answers = data_item[answer_key]

        # Add the current question
        prompt += self.marge_QA(question, candidate_answers)
        if self.enable_cot:
            prompt += f'\n{self.cot_context}'
        prompt += (
            'Please enclose your answer in parentheses. For example, (A) or (B) or (C) or (D).'
        )
        return prompt


@PromptBuilderRegistry.register('MultiChoiceAutoLabel')
class MultiChoiceAutoLabelPromptBuilder(PromptBuilder):
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


@PromptBuilderRegistry.register('MultiChoiceChinese')
class MultiChoicePromptChineseBuilder(PromptBuilder):
    def __init__(
        self,
        candidate_labels: list[str],
        multi_choice_prompt: Optional[str] = None,
        cot_context: Optional[str] = None,
        few_shot_examples: Optional[list[str]] = None,
        cot: bool = False,
    ):
        self.multi_choice_prompt = (
            multi_choice_prompt if multi_choice_prompt else '现在请回答下面的选择题。'
        )
        self.candidate_labels = candidate_labels
        self.cot_context = cot_context if cot_context else '让我们一步一步来思考。'
        self.few_shot_examples = few_shot_examples
        self.enable_cot = cot

    def marge_QA(self, question: str, candidate_answers: list[str], ground_truth: str = '') -> str:
        # Handle the ground_truth label mapping
        if ground_truth.isdigit():
            ground_truth = [self.candidate_labels[int(ground_truth)]]
        prompt = f'{question}\n'
        for label, answer in zip(self.candidate_labels, candidate_answers):
            prompt += f'({label}) {answer} '

        if ground_truth:
            answer = f'\n答案: ({ground_truth})'
        else:
            answer = ''

        return prompt + answer + '\n'

    def build_prompt(
        self,
        question: str,
        data_item: dict,
        question_key: str = 'question',
        answer_key: Union[tuple, list, str] = 'choices',
        ground_truth_key: str = 'answer',
    ) -> str:
        prompt = ''

        if self.few_shot_examples:
            prompt += '以下是带答案的多项选择题。\n'
            if isinstance(answer_key, tuple):
                answer_key = answer_key._asdict()
                answer_key_keys = list(answer_key.keys())
                for q, data_item, a in zip(
                    self.few_shot_examples[question_key],
                    self.few_shot_examples[answer_key_keys[0]],
                    self.few_shot_examples[ground_truth_key],
                ):
                    value = data_item
                    for key in answer_key_keys[1:]:
                        value = value[key]
                    prompt += self.marge_QA(q, value, str(a))
            elif isinstance(answer_key, list):
                columns = [self.few_shot_examples[key] for key in answer_key]
                for q, *answers, a in zip(
                    self.few_shot_examples[question_key],
                    *columns,
                    self.few_shot_examples[ground_truth_key],
                ):
                    prompt += self.marge_QA(q, answers, str(a))
            else:
                for q, c, a in zip(
                    self.few_shot_examples[question_key],
                    self.few_shot_examples[answer_key],
                    self.few_shot_examples[ground_truth_key],
                ):
                    prompt += self.marge_QA(q, c, str(a))

        prompt += f'{self.multi_choice_prompt}\n\n'

        if isinstance(answer_key, tuple):
            answer_key = answer_key._asdict()
            value = data_item
            for key in answer_key.keys():
                value = value[key]
            candidate_answers = value
        elif isinstance(answer_key, list):
            candidate_answers = [data_item[answer_key_single] for answer_key_single in answer_key]
        else:
            candidate_answers = data_item[answer_key]

        prompt += self.marge_QA(question, candidate_answers)
        if self.enable_cot:
            prompt += f'\n{self.cot_context}'
        return prompt


@PromptBuilderRegistry.register('Dialogue')
class DialoguePromptBuilder(PromptBuilder):
    def __init__(
        self,
        few_shot_examples: Optional[list[str]] = None,
        cot_context: Optional[str] = None,
        cot: bool = False,
    ):
        self.cot_context = cot_context if cot_context else "Let's think step by step."
        self.few_shot_examples = few_shot_examples
        self.enable_cot = cot

    def marge_QA(self, question: str, ground_truth: str = '') -> str:
        prompt = f'Question: {question}\n'
        answer = (
            f'Answer: {self.cot_context} {ground_truth}'
            if self.enable_cot
            else f'Answer: {ground_truth}'
        )
        return prompt + answer

    def build_prompt(self, input_question: str, ref_answer: str = '') -> str:
        context = ''
        if self.few_shot_examples:
            for question, answer in zip(
                self.few_shot_examples['question'], self.few_shot_examples['answer']
            ):
                context += self.marge_QA(question, answer) + '\n\n'
            context = context + '\n' if context else ''

        question = self.marge_QA(input_question)
        prompt = f'{context}{question}'
        return prompt


@PromptBuilderRegistry.register('DialogueChinese')
class DialoguePromptChineseBuilder(PromptBuilder):
    def __init__(
        self,
        few_shot_examples: Optional[list[str]] = None,
        cot_context: Optional[str] = None,
        cot: bool = False,
    ):
        self.cot_context = cot_context if cot_context else '让我们一步一步来思考。'
        self.few_shot_examples = few_shot_examples
        self.enable_cot = cot

    def marge_QA(self, question: str, ground_truth: str = '') -> str:
        prompt = f'问题: {question}\n'
        answer = (
            f'\n答案: {self.cot_context} {ground_truth}'
            if self.enable_cot
            else f'\n答案: {ground_truth}'
        )
        return prompt + answer

    def build_prompt(self, input: str) -> str:
        context = ''
        if self.few_shot_examples:
            for question, answer in zip(
                self.few_shot_examples['question'], self.few_shot_examples['answer']
            ):
                context += self.marge_QA(question, answer)
            context = context + '\n' if context else ''

        question = self.marge_QA(input)
        prompt = f'{context}{question}'
        return prompt


@PromptBuilderRegistry.register('CodesGeneration')
class CodesGenerationPromptBuilder(PromptBuilder):
    def __init__(
        self,
        few_shot_examples: Optional[list[str]] = None,
        cot_context: Optional[str] = None,
        cot: bool = False,
        language: str = 'python',
    ):
        self.cot_context = cot_context if cot_context else "Let's think step by step."
        self.few_shot_examples = few_shot_examples
        self.enable_cot = cot
        self.code_generation_prompt = (
            'The following are examples of function description (with Canonical_solution).'
        )
        self.language = language

    def build_example_prompt(self, question: str, ground_truth: str, with_answer=True):
        answer = (
            f'Canonical_solution:\n ```{self.language}\n{ground_truth}\n```' if with_answer else ''
        )
        return f'Function description:\n{question}\n{answer}'

    def build_prompt(self, question: str, ground_truth: str) -> str:
        prompt = f'{self.code_generation_prompt}\n\n'
        if self.few_shot_examples:
            for few_shot_question, few_shot_ground_truth in zip(
                self.few_shot_examples['prompt'], self.few_shot_examples['canonical_solution']
            ):
                prompt += self.build_example_prompt(few_shot_question, few_shot_ground_truth) + '\n'
            prompt += 'Now, please provide solution for the following function description:\n'

        prompt += self.build_example_prompt(question, ground_truth, with_answer=False)
        prompt += f'\nPlease provide your solution in a code block using ```{self.language}\n...\n``` format.'
        if self.enable_cot:
            prompt += f'\n{self.cot_context}'
        return prompt
