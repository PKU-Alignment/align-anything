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
import re
import json
from tqdm import tqdm
from typing import List, Dict, Any
from vllm.utils import cuda_device_count_stateless
from align_anything.evaluation.data_type import InferenceInput, InferenceOutput
from vllm import LLM, SamplingParams
from align_anything.utils.tools import requestoutput_to_dict

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

def extract_choices(prompt):
    count_pattern = r'\n\([A-Z]|[0-9]\)\s'
    num_choices = len(re.findall(count_pattern, prompt))
    
    choice_pattern = r'\(([A-Z]|[0-9])\)\s(.*?)(?=\n|$)'
    matches = re.findall(choice_pattern, prompt, re.DOTALL)
    
    choices = {f"({match[0]})": match[1].strip() for match in matches[:num_choices]}
    return choices

def save_detail(question, prompt, correct_answer, response, score, file_path, gpt_response=None):
    choices = extract_choices(prompt)
    if choices:
        record = {
            "question": question,
            "choices": choices,
            "correct_answer": correct_answer,
            "response": response,
            "score": score
        }
    else:
        record = {
            "question": question,
            "correct_answer": correct_answer,
            "response": response,
            "score": score
        }
    if gpt_response:
        record['gpt_response'] = gpt_response
    if not os.path.exists(file_path):
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump([record], file, ensure_ascii=False, indent=4)
    else:
        with open(file_path, 'r+', encoding='utf-8') as file:
            data = json.load(file)
            data.append(record)
            file.seek(0)
            json.dump(data, file, ensure_ascii=False, indent=4)
            
class BaseInferencer_vllm:
    def __init__(self, 
                 model_cfgs: Dict[str, Any],
                 vllm_cfgs,
                 **kwargs):
        self.vllm_cfgs_sp, self.vllm_cfgs_llm = vllm_cfgs.SamplingParams, vllm_cfgs.LLM
        self.model_cfgs = model_cfgs
        self.sp_n = self.vllm_cfgs_sp.n
        self.sp_top_k = self.vllm_cfgs_sp.top_k
        self.sp_top_p = self.vllm_cfgs_sp.top_p
        self.sp_temperature = self.vllm_cfgs_sp.temperature
        self.sp_max_tokens = self.model_cfgs.model_max_length
        self.sp_prompt_logprobs = self.vllm_cfgs_sp.prompt_logprobs
        self.sp_logprobs = self.vllm_cfgs_sp.logprobs

        self.llm_tokenizer_mode = self.vllm_cfgs_llm.tokenizer_mode
        self.llm_trust_remote_code = self.vllm_cfgs_llm.trust_remote_code
        self.llm_gpu_memory_utilization = self.vllm_cfgs_llm.gpu_memory_utilization
        self.llm_max_num_seqs = self.vllm_cfgs_llm.max_num_seqs
        tensor_ps = self.vllm_cfgs_llm.tensor_parallel_size
        self.llm_tensor_parallel_size = tensor_ps if tensor_ps else cuda_device_count_stateless()

        self.model_id = self.model_cfgs.model_id
        self.model_name_or_path = self.model_cfgs.model_name_or_path
        self.llm_trust_remote_code = self.model_cfgs.trust_remote_code
        self.sp_max_tokens = self.model_cfgs.model_max_length

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
            prompt_logprobs=self.sp_prompt_logprobs,
            logprobs=self.sp_logprobs
        )

        self.model = LLM(
            model=self.model_name_or_path,
            tokenizer=self.model_name_or_path,
            tokenizer_mode=self.llm_tokenizer_mode,
            trust_remote_code=self.llm_trust_remote_code,
            tensor_parallel_size=self.llm_tensor_parallel_size,
            gpu_memory_utilization=self.llm_gpu_memory_utilization,
            max_num_seqs = self.llm_max_num_seqs
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
