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


from typing import Any, Callable
from typing_extensions import TypedDict  # Python 3.10+

import torch
import transformers
from torch.utils.data import Dataset
from torchvision import transforms
from transformers.tokenization_utils import PaddingStrategy, TruncationStrategy

from align_anything.utils.multi_process import get_current_device
from align_anything.utils.template_registry import get_template_class
from align_anything.utils.tools import right_padding
from datasets import load_dataset


IGNORE_INDEX = -100

__all__ = [
    'SupervisedDataset',
    'SupervisedCollator',
    'SupervisedSample',
    'SupervisedBatch',
]


class SupervisedSample(TypedDict, total=True):
    input_ids: torch.LongTensor  # size = (L,)
    labels: torch.LongTensor  # size = (L,)
    image_pixel_values: torch.LongTensor | None  # size = (B, C, H, W)
    audio_pixel_values: torch.LongTensor | None  # size = (B, C, H, W)
    is_longer: torch.BoolTensor | None  # size = (B,)


class SupervisedBatch(TypedDict, total=True):
    input_ids: torch.LongTensor  # size = (B, L)
    labels: torch.LongTensor  # size = (B, L)
    attention_mask: torch.BoolTensor  # size = (B, L)
    image_pixel_values: torch.LongTensor | None  # size = (B, C, H, W)
    audio_pixel_values: torch.LongTensor | None  # size = (B, C, H, W)
    is_longer: torch.BoolTensor | None  # size = (B,)


class SupervisedDataset(Dataset):

    def __init__(
        self,
        path: str,
        template: str,
        tokenizer: transformers.PreTrainedTokenizer,
        processor: transformers.ProcessorMixin | transforms.Compose | None = None,
        name: str | None = None,
        size: int | None = None,
        split: str | None = None,
        subset: str | None = None,
        data_files: str | None = None,
        optional_args: list | str = [],
    ):
        super().__init__()
        assert path, f'You must set the valid datasets path! Here is {path}'
        assert template, f'You must set the valid template path! Here is {template}'
        self.path = path
        self.tokenizer = tokenizer
        self.processor = processor
        self.raw_data = load_dataset(
            path,
            name=name if name and name!="None" else None,
            split=split if split and split!="None" else None,
            data_files=data_files if data_files and data_files!="None" else None,
            *optional_args,
            trust_remote_code=True,
        )
        if size:
            self.raw_data = self.raw_data.select(range(int(size)))
        self.template = get_template_class(template)

    def preprocess(self, raw_sample: dict[str, Any]) -> SupervisedSample:
        formatted_sample = self.template.format_supervised_sample(raw_sample, self.path)
        return_dict = {}
        raw_text = ''
        if isinstance(formatted_sample['text'], list):
            raw_text = self.tokenizer.eos_token.join(formatted_sample['text'])
        elif isinstance(formatted_sample['text'], str):
            raw_text = formatted_sample['text'] + self.tokenizer.eos_token
        else:
            raise NotImplementedError
        return_dict['input_ids'] = self.tokenize(raw_text)

        formatted_prompt = formatted_sample['prompt']
        labels = return_dict['input_ids'].clone()
        # mask non-assistant input
        labels[: len(self.tokenize(formatted_prompt))] = IGNORE_INDEX
        return_dict['labels'] = labels


        if 'image' in formatted_sample.keys():
            raw_image = formatted_sample['image']
            try:
                return_dict['image_pixel_values'] = self.processor.image_processor(
                    raw_image, 
                    return_tensors='pt'
                )['pixel_values'][0]
            except Exception as e:
                return_dict['image_pixel_values'] = None
        else:
            return_dict['image_pixel_values'] = None
            
        if 'audio' in formatted_sample.keys():
            raw_audio = formatted_sample['audio']
            try:
                audio = self.processor.audio_processor(
                    raw_audio, 
                    sampling_rate = formatted_sample['sampling_rate'],
                    return_tensors='pt'
                )
                return_dict['audio_pixel_values'] = audio['input_features'][0]
                return_dict['is_longer'] = audio['is_longer'][0]
            except Exception as e:
                return_dict['audio_pixel_values'] = None
                return_dict['is_longer'] = None
        else:
            return_dict['audio_pixel_values'] = None
            return_dict['is_longer'] = None

        return return_dict

    def get_collator(self) -> Callable[[list[dict[str, torch.Tensor]]], dict[str, torch.Tensor]]:
        return SupervisedCollator(self.tokenizer.pad_token_id)

    def tokenize(
        self,
        text: str,
        add_special_tokens: bool = True,
        padding: bool | str | PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        truncation: bool | str | TruncationStrategy = TruncationStrategy.LONGEST_FIRST,
        max_length: int | None = None,
    ) -> torch.LongTensor:  # size = (L,)
        """Tokenize a text string into a tensor representation."""
        if max_length is None:
            max_length = self.tokenizer.model_max_length

        return self.tokenizer(
            text,
            add_special_tokens=add_special_tokens,
            padding=padding,
            max_length=max_length,
            truncation=truncation,
            return_tensors='pt',
        )['input_ids'][0]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Get a tokenized data sample by index."""
        raw_sample = self.raw_data[index]
        data = self.preprocess(raw_sample.copy())
        return data

    def __len__(self) -> int:
        """Get the number of samples in the dataset."""
        return len(self.raw_data)


class SupervisedCollator:

    def __init__(self, pad_token_id: int) -> None:
        """Initialize a collator."""
        self.pad_token_id = pad_token_id

    def __call__(self, samples: list[SupervisedSample]) -> SupervisedBatch:
        return_dict = {}
        current_device = get_current_device()
        return_dict['input_ids'] = right_padding(
            [sample['input_ids'] for sample in samples],
            padding_value=self.pad_token_id,
        ).to(current_device)

        return_dict['labels'] = right_padding(
            [sample['labels'] for sample in samples],
            padding_value=IGNORE_INDEX,
        ).to(current_device)

        return_dict['attention_mask'] = (
            return_dict['input_ids'].ne(self.pad_token_id).to(current_device)
        )

        if samples[0]['image_pixel_values'] is not None:
            if samples[0]['image_pixel_values'].dim() == 4:
                # init list for pixel_values
                return_dict['image_sizes'] = [ sample['image_pixel_values'].to(current_device).size(0) for sample in samples ]
                
                _pixel_values_list = []
                for sample in samples:
                    pixel_values = sample['image_pixel_values']  # size = (P, C, H, W)
                    _pixel_values_list.append(pixel_values)
                
                return_dict['image_pixel_values'] = torch.cat(_pixel_values_list, dim=0).to(current_device) 
                # size = (P1+P2+...+P_n+P1+P2+...+P_n, C, H, W) 
                
            else:
                # original code for non-patches 
                return_dict['image_pixel_values'] = torch.stack(
                    [sample['image_pixel_values'] for sample in samples]
                ).to(current_device)
        else:
            return_dict['image_pixel_values'] = None

        if samples[0]['audio_pixel_values'] is not None:
            if samples[0]['audio_pixel_values'].dim() == 4:
                # init list for pixel_values
                return_dict['audio_sizes'] = [ sample['audio_pixel_values'].to(current_device).size(0) for sample in samples ]
                
                _pixel_values_list = []
                _is_longer_list = []
                for sample in samples:
                    pixel_values = sample['audio_pixel_values']  # size = (P, C, H, W)
                    _pixel_values_list.append(pixel_values)
                    
                    is_longer = sample['is_longer']
                    _is_longer_list.append(is_longer)
                
                return_dict['audio_pixel_values'] = torch.cat(_pixel_values_list, dim=0).to(current_device) 
                # size = (P1+P2+...+P_n+P1+P2+...+P_n, C, H, W) 
                
                return_dict['is_longer'] = torch.cat(_is_longer_list, dim=0).to(current_device) 
                
            else:
                # original code for non-patches 
                return_dict['audio_pixel_values'] = torch.stack(
                    [sample['audio_pixel_values'] for sample in samples]
                ).to(current_device)
                
                return_dict['is_longer'] = torch.stack(
                    [sample['is_longer'] for sample in samples]
                ).to(current_device)
        else:
            return_dict['audio_pixel_values'] = None
            return_dict['is_longer'] = None
        return return_dict