import json
from collections import namedtuple

from eval_anything.benchmarks.text_to_text.MaliciousInstructions.utils import (
    AbsoluteHarmfulnessPredictor,
)
from eval_anything.models.base_model import BaseModel
from eval_anything.pipeline.t2t_benchmark import T2TBenchmark
from eval_anything.utils.cache_manager import CacheManager
from eval_anything.utils.data_type import EvaluationResult, InferenceInput, InferenceOutput
from eval_anything.utils.logger import EvalLogger
from eval_anything.utils.register import BenchmarkRegistry


@BenchmarkRegistry.register('MaliciousInstructions')
class MaliciousInstructions(T2TBenchmark):
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
        self.benchmark_name = 'MaliciousInstructions'
        self.benchmark_cfgs = self.get_benchmark_cfgs(self.benchmark_name)
        setup = 'redteam-osst'  # or "redteam-osst"
        self.harmfulness_predictor = AbsoluteHarmfulnessPredictor(setup, device='cuda')

    def to_InferenceInput(self, task_list: list[str]) -> dict[str, list[InferenceInput]]:
        """Convert a task list to a list of InferenceInput dict instances"""

        with open('eval-anything-local/data/MaliciousInstructions.json') as f:
            all_prompts = json.load(f)['instructions']

        inference_inputs = []
        # input_prompt = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n"

        for item in all_prompts:
            inference_inputs.append(
                InferenceInput(
                    task='MaliciousInstructions',
                    conversation=[{'role': 'user', 'content': item}],
                    ref_answer=None,
                    metadata={
                        'prompt': item,
                    },
                )
            )

        return {'MaliciousInstructions': inference_inputs}

    def to_InferenceOutput(
        self, inference_inputs: dict[str, list[InferenceInput]]
    ) -> dict[str, list[InferenceOutput]]:
        """Convert a InferenceInput dict instances to a InferenceOutput dict instances"""
        inference_outputs = super().to_InferenceOutput(inference_inputs)

        prompts = [item.metadata['prompt'] for item in inference_inputs['MaliciousInstructions']]
        responses = [item.response[0] for item in inference_outputs['MaliciousInstructions']]
        # print(responses)

        eval_scores = self.harmfulness_predictor.predict(prompts, responses)

        for output_item, eval_item in zip(inference_outputs['MaliciousInstructions'], eval_scores):
            try:
                output_item.label = eval_item
            except Exception:
                output_item.label = None
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

        all_scores = 0
        all_valid_count = 0
        for output_item in inference_outputs['MaliciousInstructions']:
            if output_item.label is not None:
                all_scores += output_item.label
                all_valid_count += 1

        mean_score = all_scores / all_valid_count

        self.display_benchmark_results(
            self.benchmark_name,
            {'MaliciousInstructions': {'Harmfulness Score': {'default': mean_score}}},
        )

        return (
            inference_outputs,
            {'MaliciousInstructions': {'Harmfulness Score': mean_score}},
            {},
        )
