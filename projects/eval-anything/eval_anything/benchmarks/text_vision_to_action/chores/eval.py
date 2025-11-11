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

from eval_anything.models.hf_vla import AccelerateVLAModel
from eval_anything.pipeline.tv2act_benchmark import TV2ACTBenchmark
from eval_anything.utils.cache_manager import CacheManager
from eval_anything.utils.data_type import EvaluationResult
from eval_anything.utils.logger import EvalLogger
from eval_anything.utils.register import BenchmarkRegistry


@BenchmarkRegistry.register('chores')
class ChoresEvalBenchmark(TV2ACTBenchmark):
    def __init__(
        self,
        eval_cfgs: namedtuple = None,
        model_cfgs: namedtuple = None,
        infer_cfgs: namedtuple = None,
        output_path: str = None,
        cache_manager: CacheManager = None,
        logger: EvalLogger = None,
    ):

        self.benchmark_name = 'chores'
        self.benchmark_cfgs = self.get_benchmark_cfgs(self.benchmark_name)
        self.data_cfgs = self.benchmark_cfgs.data_cfgs
        self.agent = AccelerateVLAModel(model_cfgs=self.benchmark_cfgs.model_cfgs)
        super().__init__(
            eval_cfgs,
            self.benchmark_cfgs.model_cfgs,
            self.benchmark_cfgs.infer_cfgs,
            self.benchmark_cfgs.output_path,
            cache_manager,
            logger,
        )

    def run(
        self,
    ) -> tuple[
        dict[str, list[EvaluationResult]], dict[str, dict[str, float]], dict[str, dict[str, float]]
    ]:
        """Run benchmark
        Args:
            task_list (list[str]): task list

        Returns:
            evaluation_details (dict[str, list[EvaluationResult]]): evaluation details
            evaluation_results (dict[str, dict[str, float]]): evaluation results
        """

        input_data = self.init_dataloader(None, self.data_cfgs)
        # input_data = dataloader.load_task_dataset()    # Input_data: list[InferenceInput]
        evaluation_results, evaluation_details = self.batch_inference(self.agent, input_data)

        if len(input_data) >= 1:
            overall_result = self.calculate_overall_metrics(evaluation_results)
        else:
            overall_result = {}

        self.display_benchmark_results(overall_result)

        self.save_benchmark_details(
            self.output_path, self.benchmark_name, evaluation_results, evaluation_details
        )
        return evaluation_details, evaluation_results, overall_result


chores = ChoresEvalBenchmark()
chores.run()
