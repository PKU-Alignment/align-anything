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
from typing import Optional, Dict, List, Any, Tuple
import pandas as pd
import numpy as np
import json
import re

from datasets import load_dataset
from tqdm import tqdm

from eval_anything.benchmarks.text_to_text.DeceptionBench.utils import (
    parse_alignment_response,
    parse_consistency_response,
    extract_cot_and_answer,
    gpt_evaluate_alignment,
    gpt_evaluate_cot_inner_alignment,
    gpt_evaluate_consistency,
    perform_advanced_analysis,
)
from eval_anything.models.base_model import BaseModel
from eval_anything.pipeline.t2t_benchmark import T2TBenchmark
from eval_anything.utils.cache_manager import CacheManager
from eval_anything.utils.cached_requests import cached_requests
from eval_anything.utils.data_type import EvaluationResult, InferenceInput, InferenceOutput
from eval_anything.utils.logger import EvalLogger
from eval_anything.utils.register import BenchmarkRegistry


@BenchmarkRegistry.register('DeceptionBench')
class DeceptionBench(T2TBenchmark):
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
        self.benchmark_name = 'DeceptionBench'
        self.benchmark_cfgs = self.get_benchmark_cfgs(self.benchmark_name)

    def to_InferenceInput(self, task_list: list[str]) -> dict[str, list[InferenceInput]]:
        """
        Create two inference inputs for each sample:
        1. inner_belief: Use inner_prompt to get internal belief
        2. outer_response: Use system_prompt + outer_prompt to get CoT and response
        Merge into the same key-value pair
        """
        dataset = load_dataset(
            path='PKU-Alignment/DeceptionBench',
            name='default',
            split='test',
        )

        all_inputs = []
        
        for idx, item in enumerate(dataset):
            # First inference: Get internal belief
            if 'inner_prompt' in item and item['inner_prompt']:
                inner_conversation = [{'role': 'user', 'content': item['inner_prompt']}]
                
                all_inputs.append(
                    InferenceInput(
                        task='DeceptionBench',
                        conversation=inner_conversation,
                        ref_answer=None,
                        metadata={
                            'sample_id': idx,
                            'inner_prompt': item['inner_prompt'],
                            'type': item.get('type', ''),
                            'setting': item.get('setting', ''),
                            'inference_type': 'inner_belief'
                        },
                    )
                )

            # Second inference: Get external response and CoT

            outer_prompt = item.get('outer_prompt', item['outer_prompt'])
            system_prompt = item.get('system_prompt', item['system_prompt'])
            format = item.get('format', item['format'])
            outer_conversation = [{
                'role': 'user', 
                'content': system_prompt + outer_prompt + format
            }]
            
            all_inputs.append(
                InferenceInput(
                    task='DeceptionBench',
                    conversation=outer_conversation,
                    ref_answer=None,
                    metadata={
                        'sample_id': idx,
                        'inner_prompt': item['inner_prompt'],
                        'outer_prompt': outer_prompt,
                        'system_prompt': item.get('system_prompt', ''),
                        'format': item.get('format', ''),
                        'type': item.get('type', ''),
                        'setting': item.get('setting', ''),
                        'inference_type': 'outer_response'
                    },
                )
            )

        return {'DeceptionBench': all_inputs}

    def to_InferenceOutput(
        self, inference_inputs: dict[str, list[InferenceInput]]
    ) -> dict[str, list[InferenceOutput]]:
        """
        Process merged inference inputs, separate two inferences and perform alignment assessment
        """
        all_inputs = inference_inputs['DeceptionBench']
        
        # Separate two types of inference inputs
        inner_inputs = [inp for inp in all_inputs if inp.metadata['inference_type'] == 'inner_belief']
        outer_inputs = [inp for inp in all_inputs if inp.metadata['inference_type'] == 'outer_response']
        
        # Perform inference separately
        inner_outputs = []
        outer_outputs = []
        
        if inner_inputs:
            # Perform internal belief inference
            inner_inference_dict = {'DeceptionBench': inner_inputs}
            inner_result_dict = super().to_InferenceOutput(inner_inference_dict)
            inner_outputs = inner_result_dict['DeceptionBench']
        
        if outer_inputs:
            # Perform external response inference
            outer_inference_dict = {'DeceptionBench': outer_inputs}
            outer_result_dict = super().to_InferenceOutput(outer_inference_dict)
            outer_outputs = outer_result_dict['DeceptionBench']
        
        # Create mapping from sample ID to internal belief
        inner_belief_map = {}
        for inp, out in zip(inner_inputs, inner_outputs):
            sample_id = inp.metadata['sample_id']
            inner_belief_map[sample_id] = out.response
        
        # Add internal belief information to external responses
        combined_outputs = []
        for inp, out in zip(outer_inputs, outer_outputs):
            sample_id = inp.metadata['sample_id']
            inner_belief = inner_belief_map.get(sample_id, '')
            cot, answer = extract_cot_and_answer(out.response)
            
            # Create new output object containing all necessary information
            combined_output = InferenceOutput(
                uuid=out.uuid,
                engine=out.engine,
                task=out.task,
                response=out.response,
                ref_answer=None
            )
            
            # Store metadata to temporary attribute (since InferenceOutput doesn't support metadata parameter)
            combined_output.metadata = {
                'sample_id': sample_id,
                'inner_belief': inner_belief,
                'inner_prompt': inp.metadata['inner_prompt'],
                'outer_prompt': inp.metadata['outer_prompt'],
                'system_prompt': inp.metadata['system_prompt'],
                'type': inp.metadata['type'],
                'setting': inp.metadata['setting'],
                'cot': cot,
                'answer': answer
            }
            
            combined_outputs.append(combined_output)
        
        # Perform advanced alignment analysis (using new evaluation functions)
        if self.benchmark_cfgs.enable_advanced_analysis == True:
            perform_advanced_analysis(self.benchmark_cfgs, combined_outputs)
        
        # Set label to is_deceptive
        for output in combined_outputs:
            if hasattr(output, 'metadata') and output.metadata:
                output.label = output.metadata.get('is_deceptive', False)
            else:
                output.label = False

        return {'DeceptionBench': combined_outputs}

    def to_EvaluationResult(
        self, task_list: list[str], inference_outputs: dict[str, list[InferenceOutput]]
    ) -> tuple[
        dict[str, list[EvaluationResult]], dict[str, dict[str, float]], dict[str, dict[str, float]]
    ]:
        """
        Convert InferenceOutput dictionary instances to evaluation details, evaluation results, and overall results
        """
        outputs = inference_outputs['DeceptionBench']
        
        # Calculate deception rate
        deceptive_count = sum(1 for output_item in outputs if output_item.label)
        deception_rate = deceptive_count / len(outputs) if outputs else 0.0
        
        evaluation_results = {'DeceptionBench': {'Deception Rate': deception_rate}}


        self.display_benchmark_results(self.benchmark_name, {'DeceptionBench': {'Deception Rate': {'default': deception_rate}}})

        return (
            inference_outputs,
            evaluation_results,
            {},
        )

    