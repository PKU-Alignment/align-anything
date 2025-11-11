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

"""
任务基类，不直接使用，而是继承后实现具体任务的逻辑

TODO
    - 代码鲁棒性
    - logger
"""

import importlib
import json
import os
from abc import ABC
from collections import namedtuple

import gradio as gr
import pandas as pd
import eval_anything.benchmarks
import eval_anything.utils.prompt_builders
from eval_anything.models.base_model import CLASS_MAP, MODEL_MAP
from eval_anything.utils.cache_manager import CacheManager
from eval_anything.utils.logger import EvalLogger
from eval_anything.utils.register import BenchmarkRegistry
from eval_anything.utils.utils import (
    custom_cfgs_to_dict,
    dict_to_namedtuple,
    namedtuple_to_dict,
    read_cfgs_from_yaml,
    update_dict,
)

__all__=['eval_anything.benchmarks', 'eval_anything.utils.prompt_builders']

class BaseTask(ABC):
    def __init__(self, overall_cfgs_name: str, **kwargs):
        self.eval_cfgs, self.model_cfgs, self.infer_cfgs = self.get_overall_configs(
            overall_cfgs_name, **kwargs
        )
        self.enable_cache = True if self.eval_cfgs.cache_dir else False
        self.output_path = self.eval_cfgs.output_dir
        self.logger = EvalLogger('Eval-Anything', log_dir=self.output_path)
        self.cache_manager = (
            CacheManager(self.eval_cfgs.cache_dir, self.logger) if self.enable_cache else None
        )
        self.model = self.load_model(self.model_cfgs, self.infer_cfgs)

    def get_overall_configs(self, overall_cfgs_name: str, **kwargs):
        """Get configs from yaml file
        Args:
            overall_cfgs_name (str): YAML filename for evaluation configurations in eval-anything/configs/.

        Returns:
            eval_cfgs (dict): eval configs, including
            model_cfgs (dict): model configs
            infer_cfgs (dict): infer configs
        """
        overall_cfgs = read_cfgs_from_yaml(yaml_relative_dir='configs', yaml_name=overall_cfgs_name)
        eval_cfgs = overall_cfgs['eval_cfgs']
        model_cfgs = overall_cfgs['model_cfgs']
        infer_cfgs = overall_cfgs['infer_cfgs']
        for k, v in kwargs.items():
            if v == '' or v is None:
                continue
            eval_cfgs = update_dict(eval_cfgs, custom_cfgs_to_dict(k, v))
            model_cfgs = update_dict(model_cfgs, custom_cfgs_to_dict(k, v))
            infer_cfgs = update_dict(infer_cfgs, custom_cfgs_to_dict(k, v))
        eval_cfgs = dict_to_namedtuple(eval_cfgs)
        model_cfgs = dict_to_namedtuple(model_cfgs)
        infer_cfgs = dict_to_namedtuple(infer_cfgs)
        return eval_cfgs, model_cfgs, infer_cfgs

    def get_benchmark_dict(self) -> dict[str, list[str]]:
        """Get benchmark name from self.task_cfgs

        Returns:
            benchmark_dict (dict[str, list[str]]): {benchmark_name: [task_name1, task_name2, ...]}
        """
        return self.eval_cfgs.benchmarks._asdict()

    def load_model(self, model_cfgs: namedtuple, infer_cfgs: namedtuple):
        backend_type = f'{infer_cfgs.infer_backend}_{model_cfgs.model_type}'
        module_name = f'eval_anything.models.{MODEL_MAP[backend_type]}'

        module = importlib.import_module(module_name)
        model_class = getattr(module, CLASS_MAP[backend_type])
        model = model_class(model_cfgs, infer_cfgs)
        return model

    def iterate_run(self) -> list[dict[str, dict[str, float]]]:
        """Iterate benchmark list and run benchmarks"""
        self.results = {}
        self.benchmark_dict = self.get_benchmark_dict()
        overall_results = {}
        for benchmark_name, task_list in self.benchmark_dict.items():
            benchmark_evaluator = BenchmarkRegistry.get_benchmark(benchmark_name)(
                self.model,
                self.eval_cfgs,
                self.model_cfgs,
                self.infer_cfgs,
                self.output_path,
                self.cache_manager,
                self.logger,
            )
            benchmark_evaluator.logger.log('info', f'Evaluating {benchmark_name}...')

            inference_inputs = benchmark_evaluator.to_InferenceInput(task_list)
            inference_outputs = benchmark_evaluator.to_InferenceOutput(inference_inputs)
            evaluation_details, evaluation_results, overall_result = (
                benchmark_evaluator.to_EvaluationResult(task_list, inference_outputs)
            )

            benchmark_evaluator.save_benchmark_details(
                self.output_path, benchmark_name, inference_inputs, evaluation_details
            )
            self.results[benchmark_name] = evaluation_results
            overall_results[benchmark_name] = overall_result
        self.display_task_results(overall_results)
        self.save_task_details(self.output_path)
        return self.results

    def shutdown_model(self):
        """Shutdown model"""
        self.model.shutdown()

    def display_task_results(self, results: dict[str, dict[str, dict[str, float]]]):
        """Display evaluation results using both console table and Gradio interface.
        Args:
            results (dict[str, dict[str, dict[str, float]]]): evaluation results
        """
        # Print table results
        for benchmark_name, result in results.items():
            if result != {}:
                self.logger.print_table(
                    title=f'Brief Report of {benchmark_name}',
                    data=result,
                    to_csv=True,
                    csv_file_name=f'{benchmark_name}_brief_report.csv',
                )

        # Launch visualization if enabled in config
        if hasattr(self.eval_cfgs, 'visualization') and self.eval_cfgs.visualization.enable:
            self._launch_visualization(results)

    def _create_visualization_data(self, results_data: dict) -> pd.DataFrame:
        """Convert results data to pandas DataFrame for visualization.
        Args:
            results_data (dict): evaluation results
        Returns:
            pd.DataFrame: formatted results data
        """
        dfs = []
        for benchmark_name, result in results_data.items():
            if result:
                df = pd.DataFrame(result).round(4)
                df['Benchmark'] = benchmark_name
                df['Modality'] = self._get_modality_type()  # Get current task's modality type
                dfs.append(df)

        if dfs:
            return pd.concat(dfs)
        return pd.DataFrame()

    def _get_modality_type(self) -> str:
        """Get the modality type of current task.
        Returns:
            str: Modality type (multimodal, text-to-text, etc.)
        """
        # Try to get modality type from class attribute
        if hasattr(self, 'modality_type'):
            return self.modality_type
        # Infer from class name, e.g., MMUndTask -> multimodal
        class_name = self.__class__.__name__.lower()
        if 'mm' in class_name:
            return 'multimodal'
        elif 't2t' in class_name:
            return 'text-to-text'
        return 'unknown'

    def _launch_visualization(self, results: dict):
        """Launch Gradio interface for results visualization.
        Args:
            results (dict): evaluation results
        """
        results_df = self._create_visualization_data(results)
        if results_df.empty:
            self.logger.warning('No results data available for visualization')
            return

        with gr.Blocks() as demo:
            gr.Markdown(f'# {self._get_modality_type().title()} Task Evaluation Results')

            with gr.Tab('Results Overview'):
                gr.Dataframe(
                    value=results_df,
                    headers=results_df.columns.tolist(),
                    row_count=(len(results_df), 'dynamic'),
                )

            with gr.Tab('Metrics Visualization'):
                metric_columns = [
                    col for col in results_df.columns if col not in ['Benchmark', 'Modality']
                ]

                for metric in metric_columns:
                    if not results_df[metric].isnull().all():
                        plot_fig = results_df.plot(
                            kind='bar', x='Benchmark', y=metric, title=f'{metric} Across Benchmarks'
                        ).figure
                        gr.Plot(value=plot_fig)

            with gr.Tab('Export'):
                gr.File(
                    value=os.path.join(self.output_path, 'evaluation_details.json'),
                    label='Download Complete Results',
                )

        # Create visualization directory and launch interface
        vis_path = os.path.join(self.output_path, 'visualization')
        os.makedirs(vis_path, exist_ok=True)

        # Get visualization server settings from config
        vis_config = self.eval_cfgs.visualization
        server_port = vis_config.port if hasattr(vis_config, 'port') else 8080
        share = vis_config.share if hasattr(vis_config, 'share') else False

        try:
            demo.launch(server_port=server_port, server_name='0.0.0.0', share=share, quiet=True)
        except Exception as e:
            self.logger.error(f'Failed to launch visualization interface: {str(e)}')

    def save_task_details(self, save_path: str):
        """Save overall evaluation results, including all benchmark results.
        Args:
            save_path (str): save path
            results (list[EvaluationResult]): evaluation results
        """
        output_dict = {
            'eval_cfgs': namedtuple_to_dict(self.eval_cfgs),
            'model_cfgs': namedtuple_to_dict(self.model_cfgs),
            'infer_cfgs': namedtuple_to_dict(self.infer_cfgs),
            'results': self.results,
        }
        with open(os.path.join(save_path, 'evaluation_details.json'), 'w') as f:
            json.dump(output_dict, f, indent=4)
