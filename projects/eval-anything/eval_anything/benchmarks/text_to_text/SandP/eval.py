import os
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from datasets import load_dataset
from tqdm import tqdm

from eval_anything.benchmarks.text_to_text.SandP.utils import (
    EVALUATE_PROMPT,
    check_eval_response,
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

    def _single_request(inference_input: InferenceInput, inference_output: InferenceOutput) -> str:
        question = inference_input.metadata['prompt']
        response = inference_output.response
        user_prompt = EVALUATE_PROMPT.format(question=question, response=response)

        messages = [{'role': 'user', 'content': user_prompt}]

        extraction = cached_requests(
            messages=messages,
            model=model,
            max_completion_tokens=1024,
            temperature=0.0,
            api_key=api_key,
            api_base=api_base,
            cache_dir=cache_dir,
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


@BenchmarkRegistry.register('SandP')
class SandP(T2TBenchmark):
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
        self.benchmark_name = 'SandP'
        self.benchmark_cfgs = self.get_benchmark_cfgs(self.benchmark_name)

    def to_InferenceInput(self, task_list: list[str]) -> dict[str, list[InferenceInput]]:
        """Convert a task list to a list of InferenceInput dict instances"""
        dataset = load_dataset(
            path=' "Lv111/SandP"',
            split='train',
        )

        inference_inputs = []
        for item in dataset:
            inference_inputs.append(
                InferenceInput(
                    task='SandP',
                    conversation=[{'role': 'user', 'content': item['prompt']}],
                    ref_answer=None,
                    metadata={
                        'prompt': item['prompt'],
                        'category': item['category'],
                    },
                )
            )

        return {'SandP': inference_inputs}

    def to_InferenceOutput(
        self, inference_inputs: dict[str, list[InferenceInput]]
    ) -> dict[str, list[InferenceOutput]]:
        """Convert a InferenceInput dict instances to a InferenceOutput dict instances"""
        inference_outputs = super().to_InferenceOutput(inference_inputs)

        jailbreak_question = inference_inputs['SandP']
        llm_response = inference_outputs['SandP']
        eval_response = gpt_evaluate(
            jailbreak_question, llm_response, cache_dir=self.eval_cfgs.cache_dir
        )
        for output_item, eval_item in zip(inference_outputs['SandP'], eval_response):
            try:
                output_item.label = check_eval_response(output_item.response, eval_item)['label']
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
        safe_helpful_count = 0
        over_refusal_count = 0
        total_count = len(inference_outputs['SandP'])
        unsure_count = 0

        type_stats = {}

        for output_item in inference_outputs['SandP']:
            question_type = output_item.metadata.get('category', 'unknown')
            if question_type not in type_stats:
                type_stats[question_type] = {'safe_helpful': 0, 'over_refusal': 0, 'total': 0, 'unsure': 0}
            
            type_stats[question_type]['total'] += 1
            
            if output_item.label == 1:  # 安全且有帮助的回答
                safe_helpful_count += 1
                type_stats[question_type]['safe_helpful'] += 1
            elif output_item.label == 0:  # 过度拒绝
                over_refusal_count += 1
                type_stats[question_type]['over_refusal'] += 1
            else:  # 无法确定
                unsure_count += 1
                type_stats[question_type]['unsure'] += 1

        safe_helpful_rate = safe_helpful_count / total_count if total_count > 0 else 0
        over_refusal_rate = over_refusal_count / total_count if total_count > 0 else 0
        unsafe_rate=1-safe_helpful_rate
        return (
            inference_outputs,
            {'SandP': {'unsafe': unsafe_rate,"over_refusal":over_refusal_rate}},
            {},
        )

