# Copyright 2024 PKU-Alignment Team. All Rights Reserved.
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
import json
import torch
import random
import numpy as np
from tqdm import tqdm
from pprint import pprint
from abc import abstractmethod
from typing import Union, List, Dict, Any, Tuple
from align_anything.utils.tools import requestoutput_to_dict
from align_anything.evaluation.data_type import InferenceInput, InferenceOutput

from vllm import LLM, SamplingParams


ACTION_GENERATION = 'generation'


class BaseGeneratorVLLM:

    action_map = {
        ACTION_GENERATION: 'generation',
    }

    def __init__(self, cfgs, vllm_cfgs):
        
        self.vllm_cfgs_sp, self.vllm_cfgs_llm = vllm_cfgs.SamplingParams, vllm_cfgs.LLM
        self.eval_cfgs, self.data_cfgs, self.model_cfgs = cfgs.default.eval_cfgs, cfgs.default.data_cfgs, cfgs.default.model_cfgs

        self.action = self.eval_cfgs.action if self.eval_cfgs.action else 'generation'
        self.num_shot = self.eval_cfgs.n_shot if self.eval_cfgs.n_shot else 0
        self.chat_template = self.model_cfgs.chat_template

        # TODO: Resolve conflicts with torch.cuda.is_available
        self.device = self.eval_cfgs.device if self.eval_cfgs.device else 'cuda'
        
        self.output_dir = self.eval_cfgs.output_dir

        self.sp_n = self.vllm_cfgs_sp.n
        self.sp_top_k = self.vllm_cfgs_sp.top_k
        self.sp_top_p = self.vllm_cfgs_sp.top_p
        self.sp_temperature = self.vllm_cfgs_sp.temperature
        self.sp_frequency_penalty = self.vllm_cfgs_sp.frequency_penalty
        self.sp_prompt_logprobs = self.vllm_cfgs_sp.prompt_logprobs
        self.sp_logprobs = self.vllm_cfgs_sp.logprobs

        self.llm_tokenizer_mode = self.vllm_cfgs_llm.tokenizer_mode
        self.llm_trust_remote_code = self.vllm_cfgs_llm.trust_remote_code
        self.llm_gpu_memory_utilization = self.vllm_cfgs_llm.gpu_memory_utilization
        self.llm_tensor_parallel_size = 2

        self.batch_size = self.eval_cfgs.batch_size if self.eval_cfgs.batch_size else 128

        self.split = self.data_cfgs.split
        self.task_dir = self.data_cfgs.task_dir
        self.candidate_labels = self.data_cfgs.candidate_labels

        self.model_id = self.model_cfgs.model_id
        self.max_length = self.model_cfgs.max_length
        self.max_new_tokens = self.model_cfgs.max_new_tokens

        self.task2details = {}
        self.detailed_filename = f'{self.model_id}_detailed'
        self.brief_filename = f'{self.model_id}_brief'

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)

        self.init_model()
        

    def init_model(self) -> None:
        
        self.samplingparams = SamplingParams(
            n=self.sp_n,
            top_k=self.sp_top_k,
            top_p=self.sp_top_p,
            temperature=self.sp_temperature,
            max_tokens=self.max_length,
            frequency_penalty=self.sp_frequency_penalty,
            prompt_logprobs=self.sp_prompt_logprobs,
            logprobs=self.sp_logprobs
        )

        self.model = LLM(
            model=self.model_cfgs.model_name_or_path,
            tokenizer=self.model_cfgs.model_name_or_path,
            tokenizer_mode=self.llm_tokenizer_mode,
            trust_remote_code=self.llm_trust_remote_code,
            tensor_parallel_size=self.llm_tensor_parallel_size,
            gpu_memory_utilization=self.llm_gpu_memory_utilization
        )
    '''
    def eval(self, data:Dict[str, List[InferenceInput]]) -> None:
        task2details = self.eval_task(data)
        
        self.update_results(task2details)
        outputs = {}
        for task, details in task2details.items():
            outputs[task] = [InferenceOutput.from_vllm_output(detail['pred']) for detail in details]
        return outputs
    
    def eval_task(self, inputs:Dict[str, List[InferenceInput]]) -> Dict[str, Any]:
        details = {}
        for task, input in inputs.items():
            details[task] = self.eval_instance(input)

        return details
        
    def eval_instance(self, instance: List[InferenceInput]) -> Dict[str, Any]:
        details_info = []
        preds = self.predict(instance)
        for i in range(len(preds)):
            detail = {}
            detail['prompt'] = instance[i].text
            details_info.append(detail)
            details_info[-1]['pred'] = preds[i]
            
        return details_info
    '''
    def update_results(self,
                       task2details:Dict[str, Dict[str, Any]]
                    )->None:
        brief_file_path = os.path.join(self.output_dir, self.brief_filename)
        detailed_file_path = os.path.join(self.output_dir, self.detailed_filename)
        
        for task, value in task2details.items():
            output_brief = []
            output_detailed = []
            
            for item in value:
                output_brief.append(requestoutput_to_dict(item, mode='brief'))
                output_detailed.append(requestoutput_to_dict(item, mode='detailed'))
                
            with open(brief_file_path + '_' + task + ".jsonl", 'w', encoding='utf-8') as file:
                for item in output_brief:
                    json_record = json.dumps(item, ensure_ascii=False)
                    file.write(json_record + '\n')

            with open(detailed_file_path + '_' + task + ".jsonl", 'w', encoding='utf-8') as file:
                for item in output_detailed:
                    json_record = json.dumps(item, ensure_ascii=False)
                    file.write(json_record + '\n')
    
    @torch.no_grad()
    def predict(self, inputs: List[InferenceInput])-> Tuple[List[str], List[Dict[str, Any]]]:
        
        action_func_name = self.action_map.get(self.action)
        if action_func_name is None:
            raise ValueError(f"Action '{self.action}' is not supported")
        action_func = getattr(self, action_func_name)
        return action_func(inputs)

    def generation(self, inputs: List[InferenceInput]):
        return self._generation(inputs)

    def _generation(self, inputs: List[InferenceInput])-> Tuple[str, Dict[str, Any]]:
        
        if inputs[0].token_ids:
            outputs = self.model.generate(prompt_token_ids=[input.token_ids for input in inputs], sampling_params=self.samplingparams)
        else:
            outputs = self.model.generate(prompts=[input.text for input in inputs], sampling_params=self.samplingparams)

        return outputs
