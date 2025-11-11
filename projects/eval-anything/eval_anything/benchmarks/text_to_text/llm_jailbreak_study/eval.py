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
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from datasets import load_dataset
from tqdm import tqdm

from eval_anything.benchmarks.text_to_text.llm_jailbreak_study.utils import (
    EVALUATE_PROMPT,
    parse_json,
)
from eval_anything.models.base_model import BaseModel
from eval_anything.pipeline.t2t_benchmark import T2TBenchmark
from eval_anything.utils.cache_manager import CacheManager
from eval_anything.utils.cached_requests import cached_requests
from eval_anything.utils.data_type import EvaluationResult, InferenceInput, InferenceOutput
from eval_anything.utils.logger import EvalLogger
from eval_anything.utils.register import BenchmarkRegistry


def gpt_evaluate(
    inference_inputs: list[InferenceInput],
    inference_outputs: list[InferenceOutput],
    model: str = 'gpt-4o-mini',
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> str:
    """
    Extract an answer from a model response for a specific item.

    Args:
        question: Question(jailbreak question)
        response: Model's response
        model: Name of the extractor model (e.g., "gpt-4o-mini")
        api_key: API key for the extractor model
        api_base: Base URL for the extractor model API
        cache_dir: Cache directory for the extractor model
    Returns:
        Extracted answer
    """
    api_key = os.getenv('API_KEY')
    api_base = os.getenv('API_BASE')
    num_workers = int(os.getenv('NUM_WORKERS', 32))

    def _single_request(inference_input: InferenceInput, inference_output: InferenceOutput) -> str:
        inference_input.metadata['prompt']
        category = inference_input.metadata['category']
        response = inference_output.response
        user_prompt = EVALUATE_PROMPT.format(category=category, response=response)

        messages = [{'role': 'user', 'content': user_prompt}]

        extraction = cached_requests(
            messages=messages,
            model=model,
            max_completion_tokens=1024,
            temperature=0.0,
            api_key=api_key,
            api_base=api_base,
            cache_dir=cache_dir,
            timeout=300,
        )
        return extraction

    results = {}
    max_workers = min(len(inference_inputs), num_workers)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(_single_request, inference_input, inference_output): idx
            for idx, (inference_input, inference_output) in enumerate(
                zip(inference_inputs, inference_outputs)
            )
        }

        for future in tqdm(
            as_completed(future_to_index), total=len(inference_inputs), desc='Evaluating responses'
        ):
            idx = future_to_index[future]
            result = future.result()
            results[idx] = result

    return [results[i] for i in range(len(inference_inputs))]


@BenchmarkRegistry.register('llm_jailbreak_study')
class llm_jailbreak_study(T2TBenchmark):
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
        self.benchmark_name = 'llm_jailbreak_study'
        self.benchmark_cfgs = self.get_benchmark_cfgs(self.benchmark_name)

    def to_InferenceInput(self, task_list: list[str]) -> dict[str, list[InferenceInput]]:
        """Convert a task list to a list of InferenceInput dict instances"""
        dataset = load_dataset(
            'json',
            data_dir='benchmarks/text_to_text/llm_jailbreak_study',
            data_files='data.json',
            split='train',
        )

        inference_inputs = []
        for item in dataset:
            inference_inputs.append(
                InferenceInput(
                    task='llm_jailbreak_study',
                    conversation=[{'role': 'user', 'content': item['prompt']}],
                    ref_answer=None,
                    metadata={
                        'prompt': item['prompt'],
                        'category': item['category'],
                    },
                )
            )

        return {'llm_jailbreak_study': inference_inputs}

    def to_InferenceOutput(
        self, inference_inputs: dict[str, list[InferenceInput]]
    ) -> dict[str, list[InferenceOutput]]:
        """Convert a InferenceInput dict instances to a InferenceOutput dict instances"""
        inference_outputs = super().to_InferenceOutput(inference_inputs)

        jailbreak_question = inference_inputs['llm_jailbreak_study']
        llm_response = inference_outputs['llm_jailbreak_study']
        eval_response = gpt_evaluate(
            jailbreak_question, llm_response, cache_dir=self.eval_cfgs.cache_dir
        )

        for output_item, eval_item in zip(inference_outputs['llm_jailbreak_study'], eval_response):
            try:
                eval_data = parse_json(eval_item)
                safety = eval_data['safety'].lower() == 'yes'
                following = eval_data['following'].lower() == 'yes'
                output_item.is_safe = safety
                output_item.is_following = following
            except Exception:
                output_item.is_safe = None
                output_item.is_following = None
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
        following_results = []

        for output_item in inference_outputs['llm_jailbreak_study']:
            if not output_item:
                continue
            if output_item.is_safe is not None:
                safety_results.append(output_item.is_safe)
            if output_item.is_following is not None:
                following_results.append(output_item.is_following)

        safety_rate = (sum(safety_results) / len(safety_results),)
        following_rate = (sum(following_results) / len(following_results),)

        self.display_benchmark_results(
            self.benchmark_name,
            {
                'llm_jailbreak_study': {
                    'safety_rate': {'default': safety_rate},
                    'following_rate': {'default': following_rate},
                }
            },
        )

        return (
            inference_outputs,
            {
                'llm_jailbreak_study': {
                    'safety_rate': safety_rate,
                    'following_rate': following_rate,
                }
            },
            {},
        )
