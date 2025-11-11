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

from datasets import load_dataset

from eval_anything.benchmarks.text_to_text.salad_bench.utils import merge_categories
from eval_anything.models.base_model import BaseModel
from eval_anything.pipeline.t2t_benchmark import T2TBenchmark
from eval_anything.utils.cache_manager import CacheManager
from eval_anything.utils.data_type import EvaluationResult, InferenceInput, InferenceOutput
from eval_anything.utils.logger import EvalLogger
from eval_anything.utils.register import BenchmarkRegistry


@BenchmarkRegistry.register('salad_bench')
class salad_bench(T2TBenchmark):
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
        self.benchmark_name = 'salad_bench'
        self.benchmark_cfgs = self.get_benchmark_cfgs(self.benchmark_name)

    def to_InferenceInput(self, task_list: list[str]) -> dict[str, list[InferenceInput]]:
        """Convert a task list to a list of InferenceInput dict instances"""
        dataset = load_dataset('OpenSafetyLab/Salad-Data', name='base_set', split='train')
        dataset = dataset.map(merge_categories)
        dataset = dataset.rename_column('question', 'prompt')
        # dataset = dataset.shuffle(seed=42).select(range(100))  # Limit to 1000 samples for testing

        inference_inputs = []
        for item in dataset:
            inference_inputs.append(
                InferenceInput(
                    task='salad_bench',
                    conversation=[{'role': 'user', 'content': item['prompt']}],
                    ref_answer=None,
                    metadata={
                        'prompt': item['prompt'],
                        'category': item['category'],
                    },
                )
            )

        return {'salad_bench': inference_inputs}

    def to_InferenceOutput(
        self, inference_inputs: dict[str, list[InferenceInput]]
    ) -> dict[str, list[InferenceOutput]]:
        """Convert a InferenceInput dict instances to a InferenceOutput dict instances"""
        inference_outputs = super().to_InferenceOutput(inference_inputs)

        jailbreak_question = inference_inputs['salad_bench']
        llm_response = inference_outputs['salad_bench']

        from .evaluator import MDJudgeEvaluator

        model_ckpt = 'OpenSafetyLab/MD-Judge-v0.1'
        md_judge = MDJudgeEvaluator(model_ckpt, use_vllm=False, use_flash_attn=False)

        questions = [input.metadata['prompt'] for input in jailbreak_question]
        responses = [output.response[0] for output in llm_response]
        eval_labels, eval_clses = md_judge.eval_unsafe_labels(questions, responses)

        for output_item, eval_label, eval_clse in zip(
            inference_outputs['salad_bench'], eval_labels, eval_clses
        ):
            try:
                safety = ['SAFE', 'UNSAFE', 'Exception'][eval_label] == 'SAFE'
                clses = eval_clse
                output_item.is_safe = safety
                output_item.clses = clses
            except Exception:
                output_item.is_safe = None
                output_item.clses = None
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

        for output_item in inference_outputs['salad_bench']:
            if not output_item:
                continue
            if output_item.is_safe is not None:
                safety_results.append(output_item.is_safe)

            following_results.append(output_item.clses)

        safety_rate = (sum(safety_results) / len(safety_results),)

        self.display_benchmark_results(
            self.benchmark_name,
            {
                'salad_bench': {
                    'safety_rate': {'default': safety_rate},
                }
            },
        )

        return (
            inference_outputs,
            {
                'salad_bench': {
                    'safety_rate': safety_rate,
                }
            },
            {},
        )
