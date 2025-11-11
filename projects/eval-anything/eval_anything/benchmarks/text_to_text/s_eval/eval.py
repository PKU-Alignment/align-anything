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

import os
import time
from collections import namedtuple

from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm

from eval_anything.benchmarks.text_to_text.s_eval.utils import (
    EN_EVALUATE_PROMPT,
    parse_json,
)
from eval_anything.models.base_model import BaseModel
from eval_anything.pipeline.t2t_benchmark import T2TBenchmark
from eval_anything.utils.cache_manager import CacheManager
from eval_anything.utils.data_type import EvaluationResult, InferenceInput, InferenceOutput
from eval_anything.utils.logger import EvalLogger
from eval_anything.utils.register import BenchmarkRegistry


def gpt_eval(questions, responses, categorys):
    api_key = os.getenv('API_KEY')
    api_base = os.getenv('API_BASE')
    client = OpenAI(api_key=api_key, base_url=api_base)

    def judger(question, response, category):
        while True:
            try:
                user_prompt = EN_EVALUATE_PROMPT.format(
                    category=category, question=question, response=response
                )
                eval_response = client.chat.completions.create(
                    model='gpt-4o',
                    messages=[
                        {
                            'role': 'user',
                            'content': user_prompt,
                        }
                    ],
                    temperature=0,
                    max_tokens=512,
                )
                break

            except Exception as err:
                print('Exception occurs when calling Chatgpt for judge:', err)
                print('Will sleep for ten seconds before retry...')
                time.sleep(10)

        return eval_response.choices[0].message.content

    eval_responses = []
    for question, response, category in tqdm(
        zip(questions, responses, categorys), total=len(questions), desc='Running Evaluation'
    ):
        eval_response = judger(question, response, category)
        eval_responses.append(eval_response)

    return eval_responses


@BenchmarkRegistry.register('s_eval')
class SEvalBenchmark(T2TBenchmark):
    def __init__(
        self,
        model: BaseModel,
        eval_cfgs: namedtuple,
        model_cfgs: namedtuple,
        infer_cfgs: namedtuple,
        output_path: str,
        cache_manager: CacheManager,
        logger: EvalLogger,
    ):
        super().__init__(
            model, eval_cfgs, model_cfgs, infer_cfgs, output_path, cache_manager, logger
        )
        self.benchmark_name = 's_eval'
        self.benchmark_cfgs = self.get_benchmark_cfgs(self.benchmark_name)

    def to_InferenceInput(self, task_list: list[str]) -> dict[str, list[InferenceInput]]:
        """Convert a task list to a list of InferenceInput dict instances"""

        dataset = load_dataset('IS2Lab/S-Eval', 'base_risk_set_en_small', split='train')
        dataset = dataset.rename_column('risk_type', 'category')
        # dataset = dataset.shuffle(seed=42).select(range(200))  # Limit to 200 samples for testing

        inference_inputs = []
        for item in dataset:
            inference_inputs.append(
                InferenceInput(
                    task='s_eval',
                    conversation=[{'role': 'user', 'content': item['prompt']}],
                    ref_answer=None,
                    metadata={
                        'prompt': item['prompt'],
                        'category': item['category'],
                    },
                )
            )

        return {'s_eval': inference_inputs}

    def to_InferenceOutput(
        self, inference_inputs: dict[str, list[InferenceInput]]
    ) -> dict[str, list[InferenceOutput]]:
        """Convert a InferenceInput dict instances to a InferenceOutput dict instances"""
        inference_outputs = super().to_InferenceOutput(inference_inputs)

        jailbreak_question = inference_inputs['s_eval']
        llm_response = inference_outputs['s_eval']

        questions = [input.metadata['prompt'] for input in jailbreak_question]
        categorys = [input.metadata['category'] for input in jailbreak_question]
        responses = [output.response[0] for output in llm_response]
        eval_responses = gpt_eval(questions, responses, categorys)

        for output_item, eval_item in zip(inference_outputs['s_eval'], eval_responses):
            try:
                eval_data = parse_json(eval_item)
                safety = eval_data['safety'].lower() == 'yes'
                output_item.is_safe = safety
            except Exception:
                output_item.is_safe = None
        return inference_outputs

    def to_EvaluationResult(
        self, task_list: list[str], inference_outputs: dict[str, list[InferenceOutput]]
    ) -> tuple[
        dict[str, list[EvaluationResult]], dict[str, dict[str, float]], dict[str, dict[str, float]]
    ]:
        """
        Convert a InferenceOutput dict instances to evaluation details, evaluation results, and
        overall results
        """
        safety_results = []

        for output_item in inference_outputs['s_eval']:
            if not output_item:
                continue
            if output_item.is_safe is not None:
                safety_results.append(output_item.is_safe)
        safety_rate = sum(safety_results) / len(safety_results)

        self.display_benchmark_results(
            self.benchmark_name, {'s_eval': {'safety_rate': {'default': safety_rate}}}
        )

        return (
            inference_outputs,
            {'s_eval': {'safety_rate': safety_rate}},
            {},
        )
