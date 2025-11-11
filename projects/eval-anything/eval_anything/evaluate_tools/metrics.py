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
from typing import Dict, List

from eval_anything.evaluate_tools.base_tools import BaseMetric, BaseTool
from eval_anything.utils.data_type import EvaluationResult
from eval_anything.utils.register import JudgeRegistry, MetricRegistry


class MetricCalculator(BaseTool):
    def __init__(self, metrics_list: List[namedtuple], judge_method: str):
        self.metrics = []
        self.metrics_args = []
        self.judge_method = judge_method
        for metric in metrics_list:
            self.metrics.append(MetricRegistry.get_metric(metric.function))
            self.metrics_args.append(metric.args)

    def apply(self, evaluation_results: List[EvaluationResult]) -> List[EvaluationResult]:
        results = {}
        for metric, args in zip(self.metrics, self.metrics_args):
            if args:
                # Convert EnhancedNameTuple to dict
                args_dict = args._asdict() if hasattr(args, '_asdict') else {}
                results[metric._registered_name] = metric(
                    evaluation_results, self.judge_method, **args_dict
                )
            else:
                results[metric._registered_name] = metric(evaluation_results, self.judge_method)
        return results

    def __call__(self, evaluation_results: List[EvaluationResult]) -> List[EvaluationResult]:
        return self.apply(evaluation_results)


@MetricRegistry.register('accuracy')
class Accuracy(BaseMetric):
    def calculate(
        self, evaluation_results: List[EvaluationResult], judge_method: str, **kwargs: None
    ) -> dict[str, float]:
        correct = {extractor: 0 for extractor in evaluation_results[0].extracted_result.keys()}
        for evaluation_result in evaluation_results:
            for extractor, extracted_result in evaluation_result.extracted_result.items():
                judge = JudgeRegistry.get_judge(judge_method)()
                if judge(extracted_result, evaluation_result.ground_truth):
                    correct[extractor] += 1
        accuracy = {
            extractor: correct[extractor] / len(evaluation_results)
            for extractor in evaluation_results[0].extracted_result.keys()
        }
        return accuracy

    def __call__(
        self, evaluation_results: List[EvaluationResult], judge_method: str, **kwargs: None
    ) -> dict[str, float]:
        return self.calculate(evaluation_results, judge_method, **kwargs)


@MetricRegistry.register('false_positive_rate')
class FalsePositiveRate(BaseMetric):
    def calculate(
        self, evaluation_results: List[EvaluationResult], judge_method: str, **kwargs: None
    ) -> dict[str, float]:
        false_positive = {
            extractor: 0 for extractor in evaluation_results[0].extracted_result.keys()
        }
        positive_label = kwargs.get('positive_label', 'true')
        judge = JudgeRegistry.get_judge(judge_method)()
        for evaluation_result in evaluation_results:
            for extractor, extracted_result in evaluation_result.extracted_result.items():
                if judge(extracted_result, positive_label) and not judge(
                    evaluation_result.ground_truth, positive_label
                ):
                    false_positive[extractor] += 1
        return {
            extractor: false_positive[extractor] / len(evaluation_results)
            for extractor in evaluation_results[0].extracted_result.keys()
        }

    def __call__(
        self, evaluation_results: List[EvaluationResult], judge_method: str, **kwargs: None
    ) -> dict[str, float]:
        return self.calculate(evaluation_results, judge_method, **kwargs)


@MetricRegistry.register('false_negative_rate')
class FalseNegativeRate(BaseMetric):
    def calculate(
        self, evaluation_results: List[EvaluationResult], judge_method: str, **kwargs: None
    ) -> dict[str, float]:
        false_negative = {
            extractor: 0 for extractor in evaluation_results[0].extracted_result.keys()
        }
        positive_label = kwargs.get('positive_label', 'true')
        judge = JudgeRegistry.get_judge(judge_method)()
        for evaluation_result in evaluation_results:
            for extractor, extracted_result in evaluation_result.extracted_result.items():
                if not judge(extracted_result, positive_label) and judge(
                    evaluation_result.ground_truth, positive_label
                ):
                    false_negative[extractor] += 1
        return {
            extractor: false_negative[extractor] / len(evaluation_results)
            for extractor in evaluation_results[0].extracted_result.keys()
        }

    def __call__(
        self, evaluation_results: List[EvaluationResult], judge_method: str, **kwargs: None
    ) -> dict[str, float]:
        return self.calculate(evaluation_results, judge_method, **kwargs)


@MetricRegistry.register('true_positive_rate')
class TruePositiveRate(BaseMetric):
    def calculate(
        self, evaluation_results: List[EvaluationResult], judge_method: str, **kwargs: None
    ) -> dict[str, float]:
        true_positive = {
            extractor: 0 for extractor in evaluation_results[0].extracted_result.keys()
        }
        positive_label = kwargs.get('positive_label', 'true')
        judge = JudgeRegistry.get_judge(judge_method)()
        for evaluation_result in evaluation_results:
            for extractor, extracted_result in evaluation_result.extracted_result.items():
                if judge(extracted_result, positive_label) and judge(
                    evaluation_result.ground_truth, positive_label
                ):
                    true_positive[extractor] += 1
        return {
            extractor: true_positive[extractor] / len(evaluation_results)
            for extractor in evaluation_results[0].extracted_result.keys()
        }

    def __call__(
        self, evaluation_results: List[EvaluationResult], judge_method: str, **kwargs: None
    ) -> dict[str, float]:
        return self.calculate(evaluation_results, judge_method, **kwargs)


@MetricRegistry.register('true_negative_rate')
class TrueNegativeRate(BaseMetric):
    def calculate(
        self, evaluation_results: List[EvaluationResult], judge_method: str, **kwargs: None
    ) -> dict[str, float]:
        true_negative = {
            extractor: 0 for extractor in evaluation_results[0].extracted_result.keys()
        }
        positive_label = kwargs.get('positive_label', 'true')
        judge = JudgeRegistry.get_judge(judge_method)()
        for evaluation_result in evaluation_results:
            for extractor, extracted_result in evaluation_result.extracted_result.items():
                if not judge(extracted_result, positive_label) and not judge(
                    evaluation_result.ground_truth, positive_label
                ):
                    true_negative[extractor] += 1
        return {
            extractor: true_negative[extractor] / len(evaluation_results)
            for extractor in evaluation_results[0].extracted_result.keys()
        }

    def __call__(
        self, evaluation_results: List[EvaluationResult], judge_method: str, **kwargs: None
    ) -> dict[str, float]:
        return self.calculate(evaluation_results, judge_method, **kwargs)


@MetricRegistry.register('precision')
class Precision(BaseMetric):
    def calculate(
        self, evaluation_results: List[EvaluationResult], judge_method: str, **kwargs: None
    ) -> dict[str, float]:
        true_positive = TruePositiveRate(evaluation_results, judge_method, **kwargs)
        false_positive = FalsePositiveRate(evaluation_results, judge_method, **kwargs)
        precision = {
            extractor: true_positive[extractor]
            / (true_positive[extractor] + false_positive[extractor])
            for extractor in evaluation_results[0].extracted_result.keys()
        }
        return precision

    def __call__(
        self, evaluation_results: List[EvaluationResult], judge_method: str, **kwargs: None
    ) -> dict[str, float]:
        return self.calculate(evaluation_results, judge_method, **kwargs)


@MetricRegistry.register('recall')
class Recall(BaseMetric):
    def calculate(
        self, evaluation_results: List[EvaluationResult], judge_method: str, **kwargs: None
    ) -> dict[str, float]:
        true_positive = TruePositiveRate(evaluation_results, judge_method, **kwargs)
        false_negative = FalseNegativeRate(evaluation_results, judge_method, **kwargs)
        recall = {
            extractor: true_positive[extractor]
            / (true_positive[extractor] + false_negative[extractor])
            for extractor in evaluation_results[0].extracted_result.keys()
        }
        return recall

    def __call__(
        self, evaluation_results: List[EvaluationResult], judge_method: str, **kwargs: None
    ) -> dict[str, float]:
        return self.calculate(evaluation_results, judge_method, **kwargs)


@MetricRegistry.register('f1_score')
class F1Score(BaseMetric):
    def calculate(
        self, evaluation_results: List[EvaluationResult], judge_method: str, **kwargs: None
    ) -> dict[str, float]:
        precision = Precision(evaluation_results, judge_method, **kwargs)
        recall = Recall(evaluation_results, judge_method, **kwargs)
        f1_score = {
            extractor: 2
            * precision[extractor]
            * recall[extractor]
            / (precision[extractor] + recall[extractor])
            for extractor in evaluation_results[0].extracted_result.keys()
        }
        return f1_score

    def __call__(
        self, evaluation_results: List[EvaluationResult], judge_method: str, **kwargs: None
    ) -> dict[str, float]:
        return self.calculate(evaluation_results, judge_method, **kwargs)


class OverallMetricCalculator(BaseTool):
    def __init__(self, metric_list: list[namedtuple]):
        self.metrics = []
        for metric in metric_list:
            if metric.function:
                self.metrics.append(MetricRegistry.get_metric(metric.function))

    def apply(self, overall_evaluation_results: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        results = {}
        for metric in self.metrics:
            results[metric._registered_name] = metric(overall_evaluation_results)
        return results

    def __call__(
        self, overall_evaluation_results: Dict[str, Dict[str, float]], **kwargs: None
    ) -> Dict[str, float]:
        return self.apply(overall_evaluation_results, **kwargs)


@MetricRegistry.register('average_across_tasks')
class AverageAcrossTasks(BaseMetric):
    def calculate(
        self, overall_evaluation_results: Dict[str, Dict[str, float]], **kwargs
    ) -> Dict[str, float]:
        metric_sums = {}
        metric_counts = {}

        for task in overall_evaluation_results.keys():
            for metric, result in overall_evaluation_results[task].items():
                metric_sums[metric] = {}
                metric_counts[metric] = {}
                for extractor, value in result.items():
                    if extractor not in metric_sums[metric].keys():
                        metric_sums[metric][extractor] = 0.0
                        metric_counts[metric][extractor] = 0
                    metric_sums[metric][extractor] += value
                    metric_counts[metric][extractor] += 1

        averages = {
            metric: {
                extractor: (
                    (metric_sums[metric][extractor] / metric_counts[metric][extractor])
                    if metric_counts[metric][extractor] > 0
                    else 'N/A'
                )
                for extractor in metric_sums[metric].keys()
            }
            for metric in metric_sums.keys()
        }

        return averages

    def __call__(
        self, overall_evaluation_results: Dict[str, Dict[str, float]], **kwargs: None
    ) -> Dict[str, float]:
        return self.calculate(overall_evaluation_results, **kwargs)
