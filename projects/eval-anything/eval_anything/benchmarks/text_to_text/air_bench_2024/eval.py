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

from eval_anything.benchmarks.text_to_text.air_bench_2024.utils import extract_content, gpt_eval
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
    model: str = 'gpt-4o',
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

    judge_prompt_list = load_dataset('stanford-crfm/air-bench-2024', 'judge_prompts', split='test')

    def get_judge_prompt(idx):
        for item in judge_prompt_list:
            if item['cate-idx'] == idx:
                return item['judge_prompt']
        raise ValueError(f'No judge prompt found for index {idx}')

    def _single_request(inference_input: InferenceInput, inference_output: InferenceOutput) -> str:
        question = inference_input.metadata['prompt']
        cate_idx = inference_input.metadata['cate_idx']
        response = inference_output.response[0]

        judge_prompt = get_judge_prompt(cate_idx)
        user_prompt = judge_prompt.replace('{{QUESTION}}', question).replace('{{ANSWER}}', response)
        messages = [{'role': 'user', 'content': user_prompt}]

        extraction = cached_requests(
            messages=messages,
            model=model,
            max_completion_tokens=512,
            temperature=0.0,
            api_key=api_key,
            api_base=api_base,
            cache_dir=cache_dir,
            timeout=300,
        )

        reason = extract_content('##short_reasoning: ', extraction)
        score = float(extract_content('##the_score: ', extraction))

        return (reason, score)

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


@BenchmarkRegistry.register('air_bench_2024')
class air_bench_2024(T2TBenchmark):
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
        self.benchmark_name = 'air_bench_2024'
        self.benchmark_cfgs = self.get_benchmark_cfgs(self.benchmark_name)

    def to_InferenceInput(self, task_list: list[str]) -> dict[str, list[InferenceInput]]:
        """Convert a task list to a list of InferenceInput dict instances"""
        # assert data_path  == "stanford-crfm/air-bench-2024"
        # assert split in ["test"], "Invalid split name"
        dataset = load_dataset('stanford-crfm/air-bench-2024', split='test')
        # dataset = dataset.shuffle(seed=42).select(range(100))  # Limit to 100 samples for testing

        inference_inputs = []
        for item in dataset:
            inference_inputs.append(
                InferenceInput(
                    task='air_bench_2024',
                    conversation=[{'role': 'user', 'content': item['prompt']}],
                    ref_answer=None,
                    metadata={
                        'cate_idx': item['cate-idx'],
                        'l2_name': item['l2-name'],
                        'l3_name': item['l3-name'],
                        'l4_name': item['l4-name'],
                        'prompt': item['prompt'],
                    },
                )
            )

        return {'air_bench_2024': inference_inputs}

    def to_InferenceOutput(
        self, inference_inputs: dict[str, list[InferenceInput]]
    ) -> dict[str, list[InferenceOutput]]:
        """Convert a InferenceInput dict instances to a InferenceOutput dict instances"""
        inference_outputs = super().to_InferenceOutput(inference_inputs)

        jailbreak_question = inference_inputs['air_bench_2024']
        llm_response = inference_outputs['air_bench_2024']

        questions = [input.metadata['prompt'] for input in jailbreak_question]
        responses = [output.response[0] for output in llm_response]
        cate_idxs = [input.metadata['cate_idx'] for input in jailbreak_question]
        eval_reasons, eval_scores = gpt_eval(questions, responses, cate_idxs)

        for output_item, eval_reason, eval_score in zip(
            inference_outputs['air_bench_2024'], eval_reasons, eval_scores
        ):
            try:
                output_item.eval_reason = eval_reason
                output_item.eval_score = eval_score
            except Exception:
                output_item.eval_reason = None
                output_item.eval_score = None
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
        eval_score_results = []
        eval_reasons_results = []

        for output_item in inference_outputs['air_bench_2024']:
            if not output_item:
                continue
            if output_item.eval_score is not None:
                eval_score_results.append(output_item.eval_score)
            if output_item.eval_reason is not None:
                eval_reasons_results.append(output_item.eval_reason)

        avg_eval_score = (sum(eval_score_results) / len(eval_score_results),)

        self.display_benchmark_results(
            self.benchmark_name,
            {
                'air_bench_2024': {
                    'eval_score': {'default': avg_eval_score},
                }
            },
        )

        return (
            inference_outputs,
            {
                'air_bench_2024': {
                    'eval_score': {'default': avg_eval_score},
                }
            },
            {},
        )
