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
import argparse
from align_anything.evaluation.inference.ds_inference import BaseInferencer_deepspeed, ListDataset, get_rank
from align_anything.evaluation.dataloader.base_dataloader import BaseDataLoader
from typing import List, Dict, Any
from align_anything.utils.tools import read_eval_cfgs, dict_to_namedtuple, update_dict, custom_cfgs_to_dict
from align_anything.utils.template_registry import get_template_class
from align_anything.evaluation.data_type import InferenceInput, InferenceOutput
from torch.nn.utils.rnn import pad_sequence
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from datasets import load_dataset, DatasetDict
import pickle
import torch
import re
import librosa
from tqdm import tqdm
import deepspeed
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
from transformers.integrations.deepspeed import HfDeepSpeedConfig
from align_anything.models.pretrained_model import load_pretrained_models

class AIRBenchDataLoader(BaseDataLoader):
    def get_task_names(self):
        if isinstance(self.data_cfgs.task, list):
            return self.data_cfgs.task
        else:
            task_names = [
            self.data_cfgs.task
            ]
            return task_names
    
    def get_answer(self, data):
        return None

    def set_fewshot_dataset(self, dataset, task: str=None):
        return None

    def build_example_prompt(self, data, with_answer=True):
        return data['question']

    def build_prompt(self, data: Dict[str, Any]) -> str:
        assert self.num_shot == 0, "AIRBench does not support few-shot learning."

        prompt = ""
        template = get_template_class(self.chat_template)
        question = [template.system_prompt + template.user_prompt.format(input=prompt + self.build_example_prompt(item, False)) + template.assistant_prompt.format(output="") for item in data]
        return question

    def preprocess(self, data, task_name):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        prompts = self.build_prompt(data)
        
        audio_data = [os.path.join(self.task_dir, 'Chat', f"{item['task_name']}_{item['dataset_name']}", item['path']) 
                      for item in data if item['task_name'] == task_name]
        
        audio_signals = [self.load_audio(audio_url) for audio_url in audio_data]
        
        inputs = [self.processor(
                    text=prompt, 
                    audios=audio_signal, 
                    return_tensors="pt", 
                    padding=True, 
                    sampling_rate=self.processor.feature_extractor.sampling_rate
                ) for prompt, audio_signal in zip(prompts, audio_signals)]
        
        return prompts, inputs
    
    def load_audio(self, audio_url):
        return librosa.load(audio_url, sr=self.processor.feature_extractor.sampling_rate)[0]

    def load_dataset(self) -> DatasetDict:
        processed_inputs = {}
        for task in tqdm(self.task_names, desc="preprocessing data"):
            dataset = load_dataset(self.task_dir, split='train', data_files='Chat/Chat_meta.json')
            prompts, inputs = self.preprocess(dataset, task)
            processed_inputs[task] = []
            for prompt, input, question_id in zip(prompts, inputs, dataset['uniq_id']):
                processed_input = InferenceInput(text=prompt, token_ids=input['input_ids'], inputs=input)
                processed_input.question_id = question_id
                processed_inputs[task].append(processed_input)
        return processed_inputs

class AIRBenchGeneratorDS(BaseInferencer_deepspeed):
    def init_model(self) -> None:
        # When using default initialization, a bug occurs where parameters are empty. Rewrite the model loading function.
        if self.infer_cfgs is not None:
            if "Qwen2-Audio" in self.model_cfgs.model_name_or_path:
                self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
                    self.model_cfgs.model_name_or_path,
                    trust_remote_code=self.model_cfgs.trust_remote_code,
                )
                self.processor = AutoProcessor.from_pretrained(self.model_cfgs.model_name_or_path)
            else:
                self.model, self.tokenizer, self.processor = load_pretrained_models(
                    self.model_cfgs.model_name_or_path,
                    model_max_length=self.model_cfgs.model_max_length,
                    padding_side='right',
                    trust_remote_code=self.model_cfgs.trust_remote_code,
                )

            self.model = deepspeed.init_inference(
                self.model,
                mp_size=torch.cuda.device_count(),  
                dtype=torch.float16,  
                replace_with_kernel_inject=True 
            )
        else:
            self.model, self.tokenizer, self.processor = load_pretrained_models(
                self.model_cfgs.model_name_or_path,
                model_max_length=self.model_cfgs.model_max_length,
                padding_side='right',
                auto_device_mapping=True,
                trust_remote_code=self.model_cfgs.trust_remote_code,
            )
        self.model.eval()
    
    def eval(self, data:Dict[str, List[InferenceInput]], eval_configs) -> Dict[str, List[InferenceOutput]]:
        os.makedirs(".cache", exist_ok=True)
        uuid_path = f".cache/{eval_configs.uuid}"
        os.makedirs(uuid_path, exist_ok=True)

        for task, input in data.items():
            task_dir = f"{uuid_path}/{task}"
            os.makedirs(task_dir, exist_ok=True)
            raw_output = self.generation(input)
            self.save_pickle(raw_output, task_dir)

    def load_data_distributed(self, inputs: List[InferenceInput]) -> List[InferenceInput]:
        dataset = ListDataset(inputs)
        
        sampler = DistributedSampler(dataset) if torch.distributed.is_initialized() else None
        
        def collate_fn(batch):
            return {
                "inputs": [b.inputs for b in batch],
                "token_ids": [b.token_ids for b in batch],
                "text": [b.text for b in batch],
                "question_id": [b.question_id for b in batch]
            }
        
        dataloader = DataLoader(
            dataset, sampler=sampler, batch_size=self.batch_size, collate_fn=collate_fn
        )
        return dataloader

    def _generation(self, inputs: List[InferenceInput]) -> List[InferenceOutput]:
        assert isinstance(inputs, list)
        dataloader = self.load_data_distributed(inputs)

        InferenceOutputs = []
        
        if self.batch_size != 1:
            import warnings
            warnings.warn(f"Batch size should be 1, current batch size is {self.batch_size}", UserWarning)
            
        for batch in tqdm(dataloader):
            local_rank = int(os.environ['LOCAL_RANK'])
            try:
                outputs = self.model.generate(
                    **batch["inputs"][0].to(f"cuda:{local_rank}"),
                    return_dict_in_generate=True, 
                    output_scores=True,
                    repetition_penalty=1.3,
                    temperature=0.1
                )
                outputs = outputs.sequences[:, batch["inputs"][0].input_ids.size(1):]
                responses = self.processor.batch_decode(outputs, skip_special_tokens=True)
                
                for i in range(self.batch_size):
                    token_ids = batch["token_ids"][i]
                    text = batch["text"][i]
                    response = responses[i]
                    inference_output = InferenceOutput.from_deepspeed_output(deepspeed_output={
                            "prompt": text,
                            "prompt_token_ids": token_ids,
                            "prompt_logprobs": None,
                            "response": response,
                            "response_token_ids": None,
                            "response_logprobs": None,
                            "raw_output":  None
                        }, store_raw=True)
                    inference_output.question_id = batch["question_id"][i]
                    InferenceOutputs.append(inference_output)
            except Exception as e:
                continue
        return InferenceOutputs

    def save_pickle(self, output_data: List[InferenceOutput], task_dir: str=None):
        cache_data = []
        for item in output_data:
            cache_data.append(
                {
                    'question_id': item.question_id,
                    'prompt_text': item.prompt,
                    'response': item.response
                }
            )
            if dist.is_initialized():
                file_path = f"{task_dir}/outputs_{get_rank()}.pkl"
            else:
                file_path = f"{task_dir}/outputs.pkl"
            
            with open(file_path, 'wb') as f:
                pickle.dump(cache_data, f, protocol=4)

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    _, unparsed_args = parser.parse_known_args()
    keys = [k[2:] for k in unparsed_args[1::2]]
    values = list(unparsed_args[2::2])
    unparsed_args = dict(zip(keys, values))
    dict_configs, infer_configs = read_eval_cfgs('air-bench', 'deepspeed')
    for k, v in unparsed_args.items():
        if v == '' or v is None:
            continue
        dict_configs = update_dict(dict_configs, custom_cfgs_to_dict(k, v))
        infer_configs = update_dict(infer_configs, custom_cfgs_to_dict(k, v))
    dict_configs = dict_to_namedtuple(dict_configs)
    model_config = dict_configs.default.model_cfgs
    eval_configs = dict_configs.default.eval_cfgs
    dataloader = AIRBenchDataLoader(dict_configs)
    test_data = dataloader.load_dataset()
    eval_module = AIRBenchGeneratorDS(model_config, infer_configs)
    eval_module.eval(test_data, eval_configs)

if __name__ == '__main__':
    main()
