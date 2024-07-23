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
from align_anything.utils.tools import read_eval_cfgs, dict_to_namedtuple, read_cfgs
from vllm import LLM, SamplingParams


ACTION_GENERATION = 'generation'

def update_results(output_dir:str,
                     brief_filename:str,
                        detailed_filename:str,
                    task2details:Dict[str, Dict[str, Any]]
                )->None:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    brief_file_path = os.path.join(output_dir, brief_filename)
    detailed_file_path = os.path.join(output_dir, detailed_filename)
    
    for task, value in task2details.items():
        output_brief = []
        output_detailed = []
        
        for item in value:
            output_brief.append(requestoutput_to_dict(item.raw_output, mode='brief'))
            output_detailed.append(requestoutput_to_dict(item.raw_output, mode='detailed'))
            
        with open(brief_file_path + '_' + task + ".jsonl", 'w', encoding='utf-8') as file:
            for item in output_brief:
                json_record = json.dumps(item, ensure_ascii=False)
                file.write(json_record + '\n')

        with open(detailed_file_path + '_' + task + ".jsonl", 'w', encoding='utf-8') as file:
            for item in output_detailed:
                json_record = json.dumps(item, ensure_ascii=False)
                file.write(json_record + '\n')
                
class BaseInferencer_vllm:
    '''
    
    '''

    action_map = {
        ACTION_GENERATION: 'generation',
    }

    def __init__(self, 
                 model_cfgs: Dict[str, Any],
                 vllm_cfgs,
                 **kwargs):
        self.vllm_cfgs_sp, self.vllm_cfgs_llm = vllm_cfgs.SamplingParams, vllm_cfgs.LLM
        self.model_cfgs = model_cfgs
        # TODO: Resolve conflicts with torch.cuda.is_available

        self.sp_n = self.vllm_cfgs_sp.n
        self.sp_top_k = self.vllm_cfgs_sp.top_k
        self.sp_top_p = self.vllm_cfgs_sp.top_p
        self.sp_temperature = self.vllm_cfgs_sp.temperature
        self.sp_max_tokens = self.vllm_cfgs_sp.max_tokens
        self.sp_frequency_penalty = self.vllm_cfgs_sp.frequency_penalty
        self.sp_prompt_logprobs = self.vllm_cfgs_sp.prompt_logprobs
        self.sp_logprobs = self.vllm_cfgs_sp.logprobs

        self.llm_tokenizer_mode = self.vllm_cfgs_llm.tokenizer_mode
        self.llm_trust_remote_code = self.vllm_cfgs_llm.trust_remote_code
        self.llm_gpu_memory_utilization = self.vllm_cfgs_llm.gpu_memory_utilization
        self.llm_tensor_parallel_size = 2

        self.model_id = self.model_cfgs.model_id
        self.model_name_or_path = self.model_cfgs.model_name_or_path
        self.llm_trust_remote_code = self.model_cfgs.trust_remote_code # rewrite this??
        self.sp_max_tokens = self.model_cfgs.model_max_length # rewrite this??

        self.task2details = {}
        self.detailed_filename = f'{self.model_id}_detailed'
        self.brief_filename = f'{self.model_id}_brief'


        self.init_model()
        

    def init_model(self) -> None:
        
        self.samplingparams = SamplingParams(
            n=self.sp_n,
            top_k=self.sp_top_k,
            top_p=self.sp_top_p,
            temperature=self.sp_temperature,
            max_tokens=self.sp_max_tokens,
            frequency_penalty=self.sp_frequency_penalty,
            prompt_logprobs=self.sp_prompt_logprobs,
            logprobs=self.sp_logprobs
        )

        self.model = LLM(
            model=self.model_name_or_path,
            tokenizer=self.model_name_or_path,
            tokenizer_mode=self.llm_tokenizer_mode,
            trust_remote_code=self.llm_trust_remote_code,
            tensor_parallel_size=self.llm_tensor_parallel_size,
            gpu_memory_utilization=self.llm_gpu_memory_utilization
        )

    def generation(self, inputs: List[InferenceInput])-> List[InferenceOutput]:
        return self._generation(inputs)

    def _generation(self, inputs: List[InferenceInput])-> List[InferenceOutput]:
        assert isinstance(inputs, list)
        if inputs[0].token_ids:
            outputs = self.model.generate(prompt_token_ids=[input.token_ids for input in inputs], sampling_params=self.samplingparams)
            for input, output in zip(inputs, outputs):
                output.prompt = input.text
        else:
            outputs = self.model.generate(prompts=[input.text for input in inputs], sampling_params=self.samplingparams)
        InferenceOutputs = [
            InferenceOutput.from_vllm_output(vllm_output=output, store_raw=True)
                for output in outputs
        ]
        return InferenceOutputs

class BaseInferencer_deepspeed:
    pass

class BaseInferencer:
    def __init__(self,
                 type,
                 model_cfgs,
                 eval_cfgs,
                 vllm_cfgs = None,
                 **kwargs
                ):
        assert type in ['vllm', 'deepspeed']
        if type == 'vllm':
            assert vllm_cfgs is not None
            self.type = type
            self.instance = BaseInferencer_vllm(model_cfgs, eval_cfgs, vllm_cfgs, **kwargs)
        else:
            pass

    def generation(self, inputs : List[InferenceInput]) -> List[InferenceOutput]:
        return self.instance.generation(inputs)