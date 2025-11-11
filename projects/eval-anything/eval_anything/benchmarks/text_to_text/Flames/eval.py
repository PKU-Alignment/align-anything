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

from eval_anything.benchmarks.text_to_text.Flames.utils import evaluate_flames
from eval_anything.models.base_model import BaseModel
from eval_anything.pipeline.t2t_benchmark import T2TBenchmark
from eval_anything.utils.cache_manager import CacheManager
from eval_anything.utils.data_type import EvaluationResult, InferenceInput, InferenceOutput
from eval_anything.utils.logger import EvalLogger
from eval_anything.utils.register import BenchmarkRegistry


@BenchmarkRegistry.register('Flames')
class FlamesBenchmark(T2TBenchmark):
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
        self.benchmark_name = 'Flames'
        self.benchmark_cfgs = self.get_benchmark_cfgs(self.benchmark_name)

    def to_InferenceInput(self, task_list: list[str]) -> dict[str, list[InferenceInput]]:
        """Convert a task list to a list of InferenceInput dict instances"""
        dataset = load_dataset(
            'json',
            data_dir='./benchmarks/text_to_text/Flames',
            data_files='Flames_1k_Chinese.jsonl',
            split='train',
        )

        inference_inputs = []
        for item in dataset:
            inference_inputs.append(
                InferenceInput(
                    task='Flames',
                    conversation=[{'role': 'user', 'content': item['prompt']}],
                    ref_answer=None,
                    metadata={
                        'prompt': item['prompt'],
                    },
                )
            )

        return {'Flames': inference_inputs}

    def to_InferenceOutput(
        self, inference_inputs: dict[str, list[InferenceInput]]
    ) -> dict[str, list[InferenceOutput]]:
        """Convert a InferenceInput dict instances to a InferenceOutput dict instances"""
        inference_outputs = super().to_InferenceOutput(inference_inputs)

        h_i, s_i, h_dim, s_dim = evaluate_flames(inference_inputs, inference_outputs)
        """
        harmless_rate: h_i
        harmless_rate_each_dim: h_dim
        harmless_score: s_i
        harmless_score_each_dim: s_dim

        """
        self.h_i, self.s_i, self.h_dim, self.s_dim = h_i, s_i, h_dim, s_dim
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

        self.display_benchmark_results(
            self.benchmark_name,
            {
                'Flames': {
                    'harmless_rate': {'default': self.h_i},
                    'harmless_rate_each_dim': self.h_dim,
                    'harmless_score': {'default': self.s_i},
                    'harmless_score_each_dim': self.s_dim,
                },
            },
        )

        return (
            inference_outputs,
            {
                'Flames': {
                    'harmless_rate': self.h_i,
                    'harmless_rate_each_dim': self.h_dim,
                    'harmless_score': self.s_i,
                    'harmless_score_each_dim': self.s_dim,
                }
            },
            {},
        )
