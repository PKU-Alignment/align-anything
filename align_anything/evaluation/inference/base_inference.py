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
import torch
from tqdm import tqdm
from typing import Dict, Any
from align_anything.models.chameleon_model_t2i import ChameleonTextToImagePipeline
from align_anything.utils.tools import requestoutput_to_dict
from diffusers import StableDiffusionPipeline, DiffusionPipeline
from transformers import ChameleonForConditionalGeneration, ChameleonProcessor, set_seed

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
            
class BaseInferencer:
    def __init__(self, 
                 model_id: str,
                 model_name_or_path: str,
                 model_max_length: str,
                 seed: str,
                 **kwargs):
        self.model_id = model_id
        self.model_name_or_path = model_name_or_path
        self.sp_max_tokens = model_max_length
        self.seed = seed

    def init_model(self, device) -> None:
        model_name_lower = self.model_id.lower()
        if "cham" in model_name_lower:
            self.model = ChameleonTextToImagePipeline(
                model_name_or_path=self.model_name_or_path,
                max_new_tokens=self.sp_max_tokens,
                seed=self.seed,
                device=device,
            )
        elif "stable-diffusion" in model_name_lower:
            self.model = StableDiffusionPipeline.from_pretrained(self.model_name_or_path).to(device)
        elif "flux" in model_name_lower or "sdxl" in model_name_lower:
            self.model = DiffusionPipeline.from_pretrained(self.model_name_or_path).to(device)
        else:
            raise ValueError(f"Model '{self.model_name_or_path}' is not supported or unknown.")
        
    def text_to_image_genenrate(self, prompt, image_path) -> None:
        model_name_lower = self.model_name_or_path.lower()
        if "cham" in model_name_lower:
            self.model.generation(prompt, image_path)
        elif "stable-diffusion" in model_name_lower:
            image = self.model(prompt).images[0]
            image.save(image_path)
        elif "flux" in model_name_lower or "sdxl" in model_name_lower:
            image = self.model(prompt).images[0]
            image.save(image_path)
        else:
            raise ValueError(f"Model '{self.model_name_or_path}' is not supported or unknown. Supported models are: chameleon, stable-diffusion, flux, sdxl.")