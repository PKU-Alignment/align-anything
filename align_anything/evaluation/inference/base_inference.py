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
import re
import torch
from tqdm import tqdm
from typing import List, Dict, Any
from torch.nn.utils.rnn import pad_sequence
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import deepspeed
from vllm.utils import cuda_device_count_stateless
from transformers.integrations.deepspeed import HfDeepSpeedConfig
from align_anything.models.pretrained_model import load_pretrained_models
from align_anything.utils.tools import requestoutput_to_dict
from align_anything.evaluation.data_type import InferenceInput, InferenceOutput
from vllm import LLM, SamplingParams
import pickle

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
        self.sp_frequency_penalty = self.vllm_cfgs_sp.frequency_penalty
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
            gpu_memory_utilization=self.llm_gpu_memory_utilization,
            distributed_executor_backend="ray",
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

class ListDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def gather_results(data, world_size):
    gathered_data = [None for _ in range(world_size)]
    dist.all_gather_object(gathered_data, data)
    return [item for sublist in gathered_data for item in sublist]

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def is_main_process():
    return get_rank() == 0

def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

class BaseInferencer_deepspeed:
    def __init__(self, 
                 model_cfgs: Dict[str, Any],
                 infer_cfgs,
                 **kwargs):
        self.infer_cfgs = infer_cfgs
        self.model_cfgs = model_cfgs

        self.model_id = self.model_cfgs.model_id
        self.model_name_or_path = self.model_cfgs.model_name_or_path
        self.llm_trust_remote_code = self.model_cfgs.trust_remote_code
        self.sp_max_tokens = self.model_cfgs.model_max_length

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = self.infer_cfgs['inference_batch_size']

        self.task2details = {}
        self.detailed_filename = f'{self.model_id}_detailed'
        self.brief_filename = f'{self.model_id}_brief'

        self.init_model()
        

    def init_model(self) -> None:
        if self.infer_cfgs is not None and self.infer_cfgs['zero_optimization']['stage'] == 3:
            self.dstchf = HfDeepSpeedConfig(self.infer_cfgs)

        if self.infer_cfgs is not None:
            self.model, self.tokenizer, self.processor = load_pretrained_models(
                self.model_cfgs.model_name_or_path,
                model_max_length=self.model_cfgs.model_max_length,
                padding_side='right',
                trust_remote_code=self.model_cfgs.trust_remote_code,
            )
            self.model, *_ = deepspeed.initialize(model=self.model, config=self.infer_cfgs)
        else:
            self.model, self.tokenizer, self.processor = load_pretrained_models(
                self.model_cfgs.model_name_or_path,
                model_max_length=self.model_cfgs.model_max_length,
                padding_side='right',
                auto_device_mapping=True,
                trust_remote_code=self.model_cfgs.trust_remote_code,
            )
        self.model.eval()
        

    def generation(self, inputs: List[InferenceInput])-> List[InferenceOutput]:
        return self._generation(inputs)

    def load_data_distributed(self, inputs: List[InferenceInput]) -> List[InferenceInput]:
        dataset = ListDataset(inputs)
        
        sampler = DistributedSampler(dataset) if torch.distributed.is_initialized() else None
        
        def collate_fn(batch):
            if not batch[0].pixel_values:
                return {
                    "pad_token_ids": pad_sequence([torch.tensor(b.token_ids) for b in batch], batch_first=True, padding_value=self.tokenizer.pad_token_id),
                    "token_ids": [b.token_ids for b in batch],
                    "text": [b.text for b in batch],
                }
            else:
                return {
                    "pad_token_ids": pad_sequence([torch.tensor(b.token_ids) for b in batch], batch_first=True, padding_value=self.tokenizer.pad_token_id),
                    'pixel_values': torch.stack([b.pixel_values for b in batch]),
                    "token_ids": [b.token_ids for b in batch],
                    "text": [b.text for b in batch],
                }
        
        dataloader = DataLoader(
            dataset, sampler=sampler, batch_size=self.batch_size, collate_fn=collate_fn
        )
        return dataloader

    def _generation(self, inputs: List[InferenceInput]) -> List[InferenceOutput]:
        assert isinstance(inputs, list)
        
        num_sequences = 4
        dataloader = self.load_data_distributed(inputs)

        InferenceOutputs = []
        
        for batch in dataloader:
            local_rank = int(os.environ['LOCAL_RANK'])
            if 'pixel_values' not in batch.keys():
                outputs = self.model.generate(
                    inputs=batch["pad_token_ids"].to(f"cuda:{local_rank}"),
                    return_dict_in_generate=True, 
                    num_return_sequences=num_sequences,
                    early_stopping=True,
                    output_scores=True,
                    num_beams=num_sequences, 
                    do_sample=True,
                    max_new_tokens=1024,
                )
            else:
                outputs = self.model.generate(
                    inputs=batch["pad_token_ids"].to(f"cuda:{local_rank}"),
                    pixel_values=batch['pixel_values'].to(f"cuda:{local_rank}"),
                    return_dict_in_generate=True, 
                    num_return_sequences=num_sequences,
                    early_stopping=True,
                    output_scores=True,
                    num_beams=num_sequences, 
                    do_sample=True,
                    max_new_tokens=1024,
                )
            transition_scores = self.model.compute_transition_scores(
                outputs['sequences'], outputs['scores'], normalize_logits=True, beam_indices=outputs['beam_indices']
            )
            responses = self.processor.batch_decode(outputs['sequences'], skip_special_tokens=True)
            
            for i in range(self.batch_size):
                token_ids = batch["token_ids"][i]
                text = batch["text"][i]
                input_length = len(token_ids)
                response = responses[i*num_sequences:(i+1)*num_sequences]
                output = outputs['sequences'][i*num_sequences:(i+1)*num_sequences, :]
                transition_score = transition_scores[i*num_sequences:(i+1)*num_sequences, :]
                InferenceOutputs.append(
                    InferenceOutput.from_deepspeed_output(deepspeed_output={
                        "prompt": text,
                        "prompt_token_ids": token_ids,
                        "prompt_logprobs": transition_score[:, :input_length],
                        "response": response,
                        "response_token_ids": output[:, input_length:],
                        "response_logprobs": transition_score[:, input_length:],
                        "raw_output":  outputs[i*num_sequences:(i+1)*num_sequences]
                    }, store_raw=True)
                )
                
        return InferenceOutputs
    
    def save_pickle(self, output_data: List[InferenceOutput]):
        os.makedirs(".cache", exist_ok=True)
        
        if dist.is_initialized():
            with open(f".cache/outputs_{get_rank()}.pkl", 'wb') as f:
                pickle.dump(output_data, f, protocol=4)
        else:
            with open(f".cache/outputs.pkl", 'wb') as f:
                pickle.dump(output_data, f, protocol=4)

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
        elif type == 'deepspeed':
            self.type = type
            self.instance = BaseInferencer_deepspeed(model_cfgs, eval_cfgs, **kwargs)

    def generation(self, inputs : List[InferenceInput]) -> List[InferenceOutput]:
        return self.instance.generation(inputs)