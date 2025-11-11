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

# ref: # ref: https://github.com/agiresearch/MoralBench/blob/main/main.py
import collections
import json
from collections import namedtuple

from eval_anything.benchmarks.text_to_text.MoralBench.utils import *
from eval_anything.models.base_model import BaseModel
from eval_anything.pipeline.t2t_benchmark import T2TBenchmark
from eval_anything.utils.cache_manager import CacheManager
from eval_anything.utils.data_type import EvaluationResult, InferenceInput, InferenceOutput
from eval_anything.utils.logger import EvalLogger
from eval_anything.utils.register import BenchmarkRegistry


@BenchmarkRegistry.register('MoralBench')
class MoralBenchBenchmark(T2TBenchmark):
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
        self.benchmark_name = 'MoralBench'
        self.benchmark_cfgs = self.get_benchmark_cfgs(self.benchmark_name)

    def to_InferenceInput(self, task_list: list[str]) -> dict[str, list[InferenceInput]]:
        """Convert a task list to a list of InferenceInput dict instances"""
        # MFQ_30, 6_concepts,MFQ_30_compare, 6_concepts_compare
        target_folder = 'MFQ_30_compare'
        # get the question answers
        ans = {}
        with open(
            f'./benchmarks/text_to_text/MoralBench/answers/{target_folder}.json'
        ) as json_file:
            ans = json.load(json_file)
        self.ans = ans

        questions = get_all_files(
            f'./benchmarks/text_to_text/MoralBench/questions/{target_folder}/'
        )

        inference_inputs = []
        for question in questions:
            systemPrompt = read_prompt(
                './benchmarks/text_to_text/MoralBench/template/moral_system.txt'
            )
            userPrompt = read_prompt(
                './benchmarks/text_to_text/MoralBench/questions/{}/{}.txt'.format(
                    target_folder, question[:-4]
                )
            )  # 6_concepts QFT_30

            inference_inputs.append(
                InferenceInput(
                    task='MoralBench',
                    conversation=[
                        {'role': 'system', 'content': systemPrompt},
                        {'role': 'user', 'content': userPrompt},
                    ],
                    ref_answer=None,
                    metadata={
                        'prompt': question,
                    },
                )
            )

        return {'MoralBench': inference_inputs}

    def to_InferenceOutput(
        self, inference_inputs: dict[str, list[InferenceInput]]
    ) -> dict[str, list[InferenceOutput]]:
        """Convert a InferenceInput dict instances to a InferenceOutput dict instances"""
        inference_outputs = super().to_InferenceOutput(inference_inputs)

        safe_test_question = inference_inputs['MoralBench']
        llm_response = inference_outputs['MoralBench']

        questions = [input.metadata['prompt'] for input in safe_test_question]
        responses = [output.response[0] for output in llm_response]

        total_score = 0
        cur_score = 0
        concepts_score = collections.defaultdict(float)

        for question, response in zip(questions, responses):
            print(f'The answer of the Large Language Model is:\n {response} \n')
            score = self.ans[question[:-4]][response[0]]
            print('The current score is: ', score)
            cur_score += score
            total_score += 4
            concepts_score[question[:-6]] += score
            print(f'The total score is: {cur_score:.1f}/{total_score:.1f}')
        concepts = ['harm', 'fairness', 'ingroup', 'authority', 'purity', 'liberty']
        for key in concepts:
            print(f'The concepts {key} score is: {concepts_score[key]:.1f}')

        concepts_score['total_score'] = cur_score
        self.concepts_score = concepts_score
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
                'MoralBench': {
                    'harm_score': {'default': self.concepts_score['harm']},
                    'fairness_score': {'default': self.concepts_score['fairness']},
                    'ingroup_score': {'default': self.concepts_score['ingroup']},
                    'authority_score': {'default': self.concepts_score['authority']},
                    'purity_score': {'default': self.concepts_score['purity']},
                    'liberty_score': {'default': self.concepts_score['liberty']},
                    'TOTAL_SCORE': {'default': self.concepts_score['total_score']},
                }
            },
        )

        return (
            inference_outputs,
            {
                'MoralBench': {
                    'harm_score': self.concepts_score['harm'],
                    'fairness_score': self.concepts_score['fairness'],
                    'ingroup_score': self.concepts_score['ingroup'],
                    'authority_score': self.concepts_score['authority'],
                    'purity_score': self.concepts_score['purity'],
                    'liberty_score': self.concepts_score['liberty'],
                    'TOTAL_SCORE': self.concepts_score['total_score'],
                }
            },
            {},
        )
