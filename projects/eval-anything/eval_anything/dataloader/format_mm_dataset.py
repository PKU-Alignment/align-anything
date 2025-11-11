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

import re
from collections import namedtuple
from typing import List

from datasets import Dataset
from tqdm import tqdm

from eval_anything.utils.data_type import InferenceInput
from eval_anything.utils.register import MMDataManagerRegistry, MMDatasetRegistry


class BaseMMDataset:
    def __init__(self, bench_cfgs: namedtuple, task: namedtuple, enable_cot: bool, num_shot: int):
        self.bench_cfgs = bench_cfgs
        self.task = task
        self.enable_cot = enable_cot
        self.num_shot = num_shot
        self.few_shot_examples = []
        self.few_shot_mm_examples = []

    def set_few_shot_examples(self, few_shot_dataset: Dataset | None):
        for item in few_shot_dataset[: self.num_shot]:
            self.few_shot_examples.append(
                {
                    'question': item[self.task.question_key],
                    'candidate_answers': item[self.task.answer_key],
                    'ground_truth': item[self.task.ground_truth_key],
                }
            )

    def _to_InferenceInput(self, dataset: Dataset):
        pass

    def __call__(self, dataset: Dataset):
        return self._to_InferenceInput(dataset)


@MMDatasetRegistry.register('mmmu')
class MMMUDataset(BaseMMDataset):
    def __init__(self, bench_cfgs: namedtuple, task: namedtuple, enable_cot: bool, num_shot: int):
        super().__init__(bench_cfgs, task, enable_cot, num_shot)

    def get_image_indice(self, text: str) -> List[int]:
        pattern = r'<image (\d+)>'
        matches = re.findall(pattern, text)
        return [int(num) for num in matches]

    def set_few_shot_examples(self, few_shot_dataset: Dataset | None):
        raise NotImplementedError('MMMU does not support few-shot learning.')

    # refer: https://github.com/MMMU-Benchmark/MMMU/blob/main/mmmu/utils/data_utils.py#L136
    def _to_InferenceInput(self, dataset: Dataset) -> List['InferenceInput']:
        """
        Convert a dataset to a list of InferenceInput objects.

        Args:
            dataset: Dataset object containing questions, options, and images

        Returns:
            List of InferenceInput objects ready for model inference
        """
        inference_inputs = []

        for item in tqdm(dataset, desc='Preparing Inference Inputs'):
            question = item['question']
            if item['question_type'] == 'multiple-choice':
                options = eval(item['options'])
                example = ''
                letter_to_option = {}

                for idx, option in enumerate(options):
                    option_letter = chr(ord('A') + idx)
                    example += f'({option_letter}) {option}\n'
                    letter_to_option[option_letter] = option

                formatted_prompt = f"{question}\n\n{example}\n\nAnswer with the option's letter from the given choices directly."
            else:
                formatted_prompt = (
                    f'{question}\n\nAnswer the question using a single word or phrase.'
                )

            image_ids = self.get_image_indice(formatted_prompt)
            images = [item[f'image_{id}'] for id in image_ids]
            image_manager = MMDataManagerRegistry.get_mm_data_manager('image')
            conversation = image_manager.prompt_to_conversation(
                user_prompt=formatted_prompt, images=images
            )

            inference_inputs.append(
                InferenceInput(
                    task=self.task.name,
                    conversation=conversation,
                    ref_answer=item['answer'],
                    metadata='image',
                )
            )
        return inference_inputs


@MMDatasetRegistry.register('mmau')
class MMAUDataset(BaseMMDataset):
    def __init__(self, bench_cfgs: namedtuple, task: namedtuple, enable_cot: bool, num_shot: int):
        super().__init__(bench_cfgs, task, enable_cot, num_shot)

    def set_few_shot_examples(self, few_shot_dataset: Dataset | None):
        raise NotImplementedError('MMAU does not support few-shot learning.')

    def _to_InferenceInput(self, dataset: Dataset) -> List['InferenceInput']:
        """
        Convert a dataset to a list of InferenceInput objects.

        Args:
            dataset: Dataset object containing questions, options, and audios

        Returns:
            List of InferenceInput objects ready for model inference
        """
        inference_inputs = []
        for item in tqdm(dataset, desc='Preparing Inference Inputs'):
            question = item['instruction']
            choices = item['choices']
            audio = item['context']['array']
            sampling_rate = item['context']['sampling_rate']

            answer_letter = re.search(r'\(([A-Z])\)', item['answer']).group(1)

            formatted_choice = ''.join([f'{choice}\n' for choice in choices])
            formatted_prompt = f"{question}\n\n{formatted_choice}\n\nAnswer with the option's letter from the given choices directly."

            audio_manager = MMDataManagerRegistry.get_mm_data_manager('audio')
            conversation = audio_manager.prompt_to_conversation(
                user_prompt=formatted_prompt, audios=audio, sample_rates=sampling_rate
            )

            inference_inputs.append(
                InferenceInput(
                    task=self.task.name,
                    conversation=conversation,
                    ref_answer=answer_letter,
                    metadata='audio',
                )
            )
        return inference_inputs


@MMDatasetRegistry.register('mathvision')
class mathvisionDataset(BaseMMDataset):
    def __init__(self, bench_cfgs: namedtuple, task: namedtuple, enable_cot: bool, num_shot: int):
        super().__init__(bench_cfgs, task, enable_cot, num_shot)

    def set_few_shot_examples(self, few_shot_dataset: Dataset | None):
        raise NotImplementedError('mathvision does not support few-shot learning.')

    # refer: https://github.com/mathllm/MATH-V/blob/main/models/Qwen-VL.py#L19
    def _to_InferenceInput(self, dataset: Dataset) -> List['InferenceInput']:
        """
        Convert a dataset to a list of InferenceInput objects.

        Args:
            dataset: Dataset object containing questions, options, and images

        Returns:
            List of InferenceInput objects ready for model inference
        """
        inference_inputs = []

        for item in dataset:
            question = item['question']
            options = ''
            if len(item['options']) > 0:
                assert len(item['options']) == 5, item
                if ''.join(item['options']) != 'ABCDE':
                    options = f"(A) {item['options'][0]}\n(B) {item['options'][1]}\n(C) {item['options'][2]}\n(D) {item['options'][3]}\n(E) {item['options'][4]}\n"
            # input = f"{question}\n{options}\nAnswer the question using a single word or phrase."
            formatted_prompt = (
                'Please solve the problem step by step and put your answer in one "\\boxed{}". If it is a multiple choice question, only one letter is allowed in the "\\boxed{}".\n'
                + f'{question}\n{options}'
            )
            formatted_prompt = (
                formatted_prompt + '<image 1>'
            )  # to fulfill the requirement of the function  prompt_to_conversation
            images = [
                item['decoded_image']
            ]  # this benchmark will only use one image for one question

            image_manager = MMDataManagerRegistry.get_mm_data_manager('image')
            conversation = image_manager.prompt_to_conversation(
                user_prompt=formatted_prompt, images=images
            )

            inference_inputs.append(
                InferenceInput(
                    task=self.task.name,
                    conversation=conversation,
                    ref_answer=item['answer'],
                    metadata='image',
                )
            )
        return inference_inputs


@MMDatasetRegistry.register('mmvu')
class MMVUDataset(BaseMMDataset):
    def __init__(self, bench_cfgs: namedtuple, task: namedtuple, enable_cot: bool, num_shot: int):
        super().__init__(bench_cfgs, task, enable_cot, num_shot)

    def set_few_shot_examples(self, few_shot_dataset: Dataset | None):
        raise NotImplementedError('MMVU does not support few-shot learning.')

    # refer: https://github.com/yale-nlp/MMVU/blob/main/utils/prepare_input.py#L11
    # refer: https://github.com/yale-nlp/MMVU/blob/main/utils/constant.py
    # TODO: Parallelizing Preprocessing Operations
    def _to_InferenceInput(self, dataset: Dataset) -> List['InferenceInput']:
        """
        Convert a dataset to a list of InferenceInput objects.

        Args:
            dataset: Dataset object containing questions, options, and videos

        Returns:
            List of InferenceInput objects ready for model inference
        """
        inference_inputs = []

        for item in tqdm(dataset, desc='Preparing Inference Inputs'):
            question = item['question']
            video = item['video']
            question_type = item['question_type']

            if question_type == 'multiple-choice':
                formatted_options = [f'({key}): {value}' for key, value in item['choices'].items()]
                options_text = '\n'.join(formatted_options)

                prompt_template = (
                    'Question: {question}\n'
                    '{options_text}\n\n'
                    'Answer the given multiple-choice question step by step. '
                    'Begin by explaining your reasoning process clearly. '
                    'Conclude by stating the final answer using the following format: '
                    "'Therefore, the final answer is: (LETTER)' (without quotes), "
                    'where (LETTER) is one of the options. Think step by step before answering.'
                )

                formatted_prompt = prompt_template.format(
                    question=question, options_text=options_text
                )

            elif question_type == 'open-ended':
                prompt_template = (
                    '{question}\n\n'
                    'Answer the given question step by step. '
                    'Begin by explaining your reasoning process clearly. '
                    'Conclude by stating the final answer using the following format: '
                    "'Therefore, the final answer is: Answer: $$ANSWER' (without quotes), "
                    'where $$ANSWER is the final answer of the question. '
                    'Think step by step before answering.'
                )

                formatted_prompt = prompt_template.format(question=question)

            else:
                raise ValueError(f'Invalid question type: {question_type}')

            video_manager = MMDataManagerRegistry.get_mm_data_manager('video')
            conversation = video_manager.prompt_to_conversation(
                user_prompt=formatted_prompt, videos=video, fps=1
            )

            inference_inputs.append(
                InferenceInput(
                    task=self.task.name,
                    conversation=conversation,
                    ref_answer=item['answer'],
                    metadata='video',
                )
            )

            if len(inference_inputs) > 10:
                break

        return inference_inputs
