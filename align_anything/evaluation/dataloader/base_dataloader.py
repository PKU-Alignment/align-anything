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
import numpy as np
from PIL import Image
from tqdm import tqdm
from pprint import pprint
from abc import abstractmethod
from torch.utils.data import Dataset
from torchvision import transforms
from typing import Union, List, Dict, Any, Tuple
from datasets import load_dataset, DatasetDict
from align_anything.evaluation.data_type import InferenceInput

from transformers import AutoProcessor, AutoTokenizer, AutoImageProcessor

ACTION_GENERATION = 'generation'

class BaseDataLoader:
    action_map = {
        ACTION_GENERATION: 'generation',
    }

    def __init__(self, cfgs):
        self.eval_cfgs, self.data_cfgs, self.model_cfgs = cfgs.default.eval_cfgs, cfgs.default.data_cfgs, cfgs.default.model_cfgs
        self.action = self.eval_cfgs.action if self.eval_cfgs.action else 'generation'
        self.num_shot = self.eval_cfgs.n_shot if self.eval_cfgs.n_shot else 0
        self.cot = self.eval_cfgs.cot if self.eval_cfgs.cot else False
        self.chat_template = self.model_cfgs.chat_template
        self.model_name_or_path = self.model_cfgs.model_name_or_path
        self.split = self.data_cfgs.split
        self.task_dir = self.data_cfgs.task_dir
        self.candidate_labels = self.data_cfgs.candidate_labels
        self.task_names = self.get_task_names()
        self.init_tokenizer()

    def init_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, trust_remote_code=True)

        try:
            self.processor = AutoProcessor.from_pretrained(self.model_name_or_path, trust_remote_code=True)
            setattr(self.processor, 'tokenizer', self.tokenizer)
        except:
            self.processor = None
            
        try:
            self.image_processor = AutoImageProcessor.from_pretrained(self.model_name_or_path, trust_remote_code=True)
        except:
            self.image_processor = None

    def load_dataset(self) -> DatasetDict:
        processed_inputs = {}
        for task in self.task_names:
            dataset = load_dataset(self.task_dir, task)
            self.few_shot_data = self.set_fewshot_dataset(dataset, task)
            prompts, token_ids = self.preprocess(dataset)
            processed_inputs[task] = [InferenceInput(text=prompt, token_ids=token_id) for prompt, token_id in zip(prompts, token_ids['input_ids'])]
        return processed_inputs

    def preprocess(self, data):
        prompts = self.build_prompt(data[self.split])
        token_ids = self.tokenizer(prompts)
        return prompts, token_ids

    def set_fewshot_dataset(self, dataset, task: str=None):
        return None
    
    @abstractmethod
    def get_task_names(self)-> List[str]:
        raise NotImplementedError

    @abstractmethod
    def build_example_prompt(self, data, with_answer: bool=False):
        raise NotImplementedError

    @abstractmethod
    def build_prompt(self, data: Dict[str, Any])-> str:
        raise NotImplementedError

    @abstractmethod
    def get_answer(self, data):
        raise NotImplementedError

class CustomImageDataset(Dataset):
    def __init__(self, image_folder):
        self.image_folder = image_folder
        self.image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(('png', 'jpg', 'jpeg'))]
        
        self.transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        return image