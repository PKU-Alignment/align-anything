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

from abc import ABC, abstractmethod
from collections import namedtuple
from typing import Dict, List

from eval_anything.evaluate_tools.metrics import MetricCalculator, OverallMetricCalculator
from eval_anything.models.base_model import BaseModel
from eval_anything.utils.cache_manager import CacheManager
from eval_anything.utils.data_type import EvaluationResult, InferenceInput, InferenceOutput
from eval_anything.utils.logger import EvalLogger
from eval_anything.utils.register import AnswerExtractorRegistry, BenchmarkRegistry
from eval_anything.utils.utils import dict_to_namedtuple, pair_data_via_uuid, read_cfgs_from_yaml


BENCHMARK_MODALITY_MAP = {
    'gsm8k': 'text_to_text',
    'mmlu': 'text_to_text',
    'truthfulqa': 'text_to_text',
    'arc': 'text_to_text',
    'cmmlu': 'text_to_text',
    'mmlupro': 'text_to_text',
    'ceval': 'text_to_text',
    'humaneval': 'text_to_text',
    'agieval': 'text_to_text',
    'mmmu': 'text_image_to_text',
    'mathvision': 'text_image_to_text',
    'mmau': 'text_audio_to_text',
    'mmvu': 'text_video_to_text',
    'doanythingnow': 'text_to_text',
    'donotanswer': 'text_to_text',
    'harmbench': 'text_to_text',
    'redeval': 'text_to_text',
    'hexphi': 'text_to_text',
    'latentjailbreak': 'text_to_text',
    'maliciousinstruct': 'text_to_text',
    'maliciousinstructions': 'text_to_text',
    'harmfulq': 'text_to_text',
    'gptfuzzer': 'text_to_text',
    'llm_jailbreak_study': 'text_to_text',
    'jbb_behaviors': 'text_to_text',
    'salad_bench': 'text_to_text',
    'air_bench_2024': 'text_to_text',
    'aegis_aicontent_safety_dataset': 'text_to_text',
    's_eval': 'text_to_text',
    'fakealignment': 'text_to_text',
    'flames': 'text_to_text',
    'xsafety': 'text_to_text',
    'jade_db': 'text_to_text',
    'moralbench': 'text_to_text',
    'chores': 'text_vision_to_action',
    # 来自 my-dev 的新增映射
    'advbench': 'text_to_text',
    'beavertails': 'text_to_text',
    'cona': 'text_to_text',
    'cyberattackassistance': 'text_to_text',
    'cdialbias': 'text_to_text',
    'bbq': 'text_to_text',
    'confaide': 'text_to_text',
    'decodingtrust': 'text_to_text',
    'anthropics': 'text_to_text',
    'chores': 'text_vision_to_action',
    'deceptionbench': 'text_to_text',
}


@BenchmarkRegistry.register('base')
class BaseBenchmark(ABC):

    def __init__(
        self,
        model: BaseModel,
        eval_cfgs: namedtuple,
        model_cfgs: namedtuple,
        infer_cfgs: namedtuple,
        output_path: str,
        cache_manager: CacheManager = None,
        logger: EvalLogger = None,
    ):
        self.model = model
        self.eval_cfgs = eval_cfgs
        self.model_cfgs = model_cfgs
        self.infer_cfgs = infer_cfgs
        self.output_path = output_path
        self.logger = logger
        self.cache_manager = cache_manager
        self.enable_cache = True if cache_manager else False
        self.benchmark_name = 'base'

    def get_benchmark_cfgs(self, benchmark_name: str) -> dict:
        """Get benchmark configs from yaml file in benchmark folder
        Args:
            benchmark_name (str): benchmark name

        Returns:
            benchmark_cfgs (dict): benchmark configs, including
        """
        benchmark_cfgs = read_cfgs_from_yaml(
            yaml_relative_dir=f'benchmarks/{BENCHMARK_MODALITY_MAP[benchmark_name.lower()]}/{benchmark_name}',
            yaml_name=f'configs.yaml',
        )
        benchmark_cfgs = dict_to_namedtuple(benchmark_cfgs)
        return benchmark_cfgs

    @abstractmethod
    def init_dataloader(self, eval_cfgs: namedtuple, benchmark_cfgs: namedtuple):
        """Load evaluation dataset
        Args:
            eval_cfgs (namedtuple): evaluation configs
            benchmark_cfgs (namedtuple): benchmark configs
            logger (EvalLogger): logger

        Returns:
            dataloader (BaseDataLoader): dataloader
        """

    def to_InferenceInput(self, task_list: list[str]) -> dict[str, list[InferenceInput]]:
        """Convert a task list to a InferenceInput dict instances"""
        dataloader = self.init_dataloader(self.eval_cfgs, self.benchmark_cfgs)
        return dataloader.load_dataset(task_list)

    def to_InferenceOutput(
        self, inference_inputs: dict[str, list[InferenceInput]]
    ) -> dict[str, list[InferenceOutput]]:
        """Convert a InferenceInput dict instances to a InferenceOutput dict instances"""
        return self.batch_inference(self.model, inference_inputs)

    def to_EvaluationResult(
        self, task_list: list[str], inference_outputs: dict[str, list[InferenceOutput]]
    ) -> tuple[
        dict[str, list[EvaluationResult]], dict[str, dict[str, float]], dict[str, dict[str, float]]
    ]:
        """Convert a InferenceOutput dict instances to evaluation details, evaluation results, and overall results"""
        evaluation_details = {}
        evaluation_results = {}
        if task_list == []:
            task_list = self.benchmark_cfgs.dataset.default_task_list

        for task in task_list:
            if self.benchmark_cfgs.judge_method is not None:
                evaluation_details[task], evaluation_results[task] = self.calculate_metrics(
                    self.benchmark_name,
                    inference_outputs[task],
                    self.benchmark_cfgs.answer_extractor,
                    self.benchmark_cfgs.metrics,
                    self.benchmark_cfgs.judge_method,
                )
            else:
                evaluation_details[task], evaluation_results[task] = self.calculate_metrics(
                    self.benchmark_name,
                    inference_outputs[task],
                    self.benchmark_cfgs.answer_extractor,
                    self.benchmark_cfgs.metrics,
                )

        if len(task_list) == 1:
            overall_result = {}
        else:
            overall_result = self.calculate_overall_metrics(
                self.benchmark_cfgs.overall_metrics, result=evaluation_results
            )

        self.display_benchmark_results(self.benchmark_name, evaluation_results)
        if overall_result != {}:
            self.display_benchmark_results(self.benchmark_name, overall_result)

        return evaluation_details, evaluation_results, overall_result

    def batch_inference(
        self, model, input_dict: dict[str, list[InferenceInput]]
    ) -> dict[str, list[InferenceOutput]]:
        # TODO 需要支持多轮
        """Model inference. Support multi-round inference.
        Args:
            input_data (list[InferenceInput]): input data

        Returns:
            inference_outputs (list[InferenceOutput]): inference outputs
        """

        input_list = [item for _, inference_input in input_dict.items() for item in inference_input]

        input_batch_size = 500
        input_data_batches = [
            input_list[i : i + input_batch_size]
            for i in range(0, len(input_list), input_batch_size)
        ]
        inference_outputs = []
        for input_data_batch in input_data_batches:
            if self.enable_cache:
                cache_path, cache_exist = self.cache_manager.get_cache_path(
                    self.model_cfgs, self.infer_cfgs, input_data_batch
                )
                if cache_exist:
                    inference_outputs.extend(self.cache_manager.load(cache_path))
                else:
                    batch_outputs = self.model_inference(model, input_data_batch)
                    inference_outputs.extend(batch_outputs)
                    self.cache_manager.save(cache_path, batch_outputs)
            else:
                inference_outputs.extend(self.model_inference(model, input_data_batch))

        outputs = {task: [] for task in input_dict.keys()}
        for output in inference_outputs:
            outputs[output.task].append(output)
        return outputs

    def model_inference(self, model, input_data: list[InferenceInput]):
        # TODO 需要支持多轮
        """Model inference. Support multi-round inference.
        Args:
            input_data (list[InferenceInput]): input data

        Returns:
            inference_outputs (list[InferenceOutput]): inference outputs
        """
        return model.generation(input_data)

    def calculate_overall_metrics(
        self, metric_list: list[namedtuple], result: dict[str, dict[str, dict[str, float]]] = None
    ):
        """Calculate overall metrics
        Args:
            metric_list (list[namedtuple]): metric list
            result (dict[str, dict[str, dict[str, float]]]): evaluation results. {task: {metric: {extractor: score}}}

        Returns:
            overall_metrics (dict[str, dict[str, dict[str, float]]]): overall metrics {overall_metric: {metric: {extractor: score}}}
        """
        overall_metric_calculator = OverallMetricCalculator(metric_list)
        return overall_metric_calculator(result)

    def calculate_metrics(
        self,
        benchmark_name: str,
        inference_outputs: list[InferenceOutput],
        answer_extractors: list[namedtuple],
        metrics_list: list[dict],
        judge_method: str = 'judge_equal',
    ):
        """Calculate metrics
        Args:
            benchmark_name (str): benchmark name
            inference_outputs (list[InferenceOutput]): inference outputs
            evaluate_tools (list[str]): evaluate tool list
            metrics_list (list[dict]): metrics list
            judge_method (str): judge method

        Returns:
            evaluation_details (dict[str, list[EvaluationResult]]): evaluation details
            evaluation_results (dict[str, dict[str, float]]): evaluation results
        """
        output_text = [item.response[0] for item in inference_outputs]
        extracted_results = {}
        for answer_extractor in answer_extractors:
            if answer_extractor.function is not None:
                extractor_class = AnswerExtractorRegistry.get_extractor(answer_extractor.function)
                extractor = extractor_class(**(answer_extractor.args._asdict()))
                extracted_results[answer_extractor.name] = extractor.apply(output_text)
            else:
                extracted_results[answer_extractor.name] = output_text

        evaluation_details = []
        ref_answers = [item.ref_answer for item in inference_outputs]
        for inference_output, ref_answer, index in zip(
            inference_outputs, ref_answers, range(len(inference_outputs))
        ):
            extracted_result = {
                extractor: extracted_results[extractor][index]
                for extractor in extracted_results.keys()
            }
            evaluation_details.append(
                EvaluationResult(
                    benchmark_name,
                    inference_output,
                    extracted_result,
                    ref_answer,
                    inference_output.uuid,
                )
            )
        metric_calculator = MetricCalculator(metrics_list, judge_method)
        evaluation_results = metric_calculator(evaluation_details)
        return evaluation_details, evaluation_results

    def calculate_overall_metrics(
        self, overall_metrics: list[namedtuple], result: dict[str, dict[str, float]]
    ) -> dict[str, float]:
        """Calculate overall metrics
        Args:
            overall_metrics (list[namedtuple]): overall metrics
            result (dict[str, dict[str, float]]): evaluation results
        """
        overall_metric_calculator = OverallMetricCalculator(overall_metrics)
        overall_result = overall_metric_calculator(result)
        return overall_result

    def get_ref_answer(
        self, input_data: list[InferenceInput], inference_outputs: list[InferenceOutput]
    ) -> list[any]:
        """Get reference answers from input data
        Args:
            input_data (list[InferenceInput]): input data
            inference_outputs (list[InferenceOutput]): inference outputs

        Returns:
            ref_answers (list[any]): reference answers
        """
        paired_data = pair_data_via_uuid(input_data, inference_outputs)
        ref_answer = [item.ref_answer for item, _ in paired_data]
        return ref_answer

    def display_benchmark_results(self, benchmark_name: str, result: Dict[str, Dict[str, float]]):
        """Display single benchmark results in command line.
        Args:
            benchmark_name (str): benchmark name
            result (Dict[str, Dict[str, float]]): evaluation result
        """
        result_to_display = {}
        for task, task_result in result.items():
            result_to_display[task] = {}
            for metric, metric_result in task_result.items():
                metric_result = max(metric_result.values())
                result_to_display[task][metric] = metric_result
        self.logger.print_table(
            title=f'{benchmark_name} Benchmark',
            data=result_to_display,
            to_csv=True,
            csv_file_name=f'{benchmark_name}_results.csv',
        )
        self.logger.log('info', '++++++++++++++++++++++++++++++++++++')
        self.logger.log('info', f'Benchmark: {benchmark_name}')
        self.logger.log('info', f'model_id: {self.model_cfgs.model_id},')
        self.logger.log('info', f'num_fewshot: {self.eval_cfgs.n_shot._asdict()[benchmark_name]},')
        self.logger.log(
            'info', f'chain_of_thought: {self.eval_cfgs.cot._asdict()[benchmark_name]},'
        )
        self.logger.log('info', '++++++++++++++++++++++++++++++++++++')

    @abstractmethod
    def save_benchmark_details(
        self,
        save_path: str,
        benchmark_name: str,
        inputs: Dict[str, List[InferenceInput]],
        results: Dict[str, List[EvaluationResult]],
    ):
        """Save evaluation result and config file of single benchmark.
        Args:
            save_path (str): save path
            benchmark_name (str): benchmark name
            inputs (dict[str, List[InferenceInput]]): evaluation inputs
            results (dict[str, List[EvaluationResult]]): evaluation results
        """
