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

from typing import Any, Callable, Dict, Optional
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
from transformers import DataCollatorForSeq2Seq, PreTrainedTokenizer, ProcessorMixin

IGNORE_INDEX = -100

__all__ = [
    'SupervisedDataset',
    'SupervisedTokenizedDataset',
    'SupervisedCollator',
    'SupervisedSample',
    'SupervisedBatch',
]


class SupervisedSample(TypedDict, total=True):
    input_ids: torch.LongTensor  # size = (L,)
    labels: torch.LongTensor  # size = (L,)
    pixel_values: torch.LongTensor | None  # size = (B, C, H, W)


class SupervisedBatch(TypedDict, total=True):
    input_ids: torch.LongTensor  # size = (B, L)
    labels: torch.LongTensor  # size = (B, L)
    attention_mask: torch.BoolTensor  # size = (B, L)
    pixel_values: torch.LongTensor | None  # size = (B, C, H, W)


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
        self.tokenizer = tokenizer
        self.processor = processor
        self.raw_data = load_dataset(
            path,
            split=split,
            data_files=data_files,
            subset=subset,
            *optional_args,
            trust_remote_code=True,
        )
        if size:
            self.raw_data = self.raw_data.select(range(int(size)))
        self.template = get_template_class(template)

    def preprocess(self, raw_sample: dict[str, Any]) -> SupervisedSample:
        formatted_sample = self.template.format_sample(raw_sample)
        return_dict = {}
        if formatted_sample['mode'] == 'TG':
            input_kwargs = dict(
                mode='G',
                ratio="1:1",
                image_area=518400,
                return_tensors="pt",
            )
            inputs = self.processor(text=formatted_sample['input_text'], **input_kwargs)
            output_kwargs = dict(
                mode='TG',
                ratio="1:1",
                image_area=518400,
                return_tensors="pt",
            )
            outputs = self.processor(text=formatted_sample['input_text'], output_image=formatted_sample['output_image'], **output_kwargs)
            full_input_ids = outputs['input_ids'][0]
            labels = full_input_ids.clone()
            prompt_input_ids = inputs['input_ids'][0]
            labels[:len(prompt_input_ids)] = IGNORE_INDEX
            return_dict['input_ids'] = full_input_ids
            return_dict['labels'] = labels
            return return_dict
        
        elif formatted_sample['mode'] == 'TU':
            inputs = self.processor(
                text=formatted_sample['input_text'],
                image=formatted_sample['input_image'],
                mode='U',
                padding_side="left",
                padding="longest",
                return_tensors="pt",
            )
            outputs = self.processor(
                text=formatted_sample['input_text'],
                image=formatted_sample['input_image'],
                output_text=formatted_sample['output_text'],
                mode='TU',
                padding_side="left",
                padding="longest",
                return_tensors="pt",
            )
            full_input_ids = outputs['input_ids'][0]
            labels = full_input_ids.clone()
            prompt_input_ids = inputs['input_ids'][0]
            labels[:len(prompt_input_ids)] = IGNORE_INDEX
            return_dict['input_ids'] = full_input_ids
            return_dict['labels'] = labels
            return return_dict

        else:
            raise ValueError(f"Invalid mode: {formatted_sample['mode']}")

    def get_collator(self) -> Callable[[list[dict[str, torch.Tensor]]], dict[str, torch.Tensor]]:
        return SupervisedCollator(self.tokenizer, self.processor)

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


class SupervisedTokenizedDataset(Dataset):

    def __init__(
        self,
        path: str,
        template: str | None = None,
        tokenizer: transformers.PreTrainedTokenizer | None = None,
        processor: transformers.ProcessorMixin | transforms.Compose | None = None,
        size: int | None = None,
        name: str | None = None,
        split: str | None = None,
        subset: str | None = None,
        data_files: str | None = None,
        optional_args: list | str = [],
    ):
        super().__init__()
        assert path, f'You must set the valid datasets path! Here is {path}'
        assert template, f'You must set the valid template path! Here is {template}'
        self.tokenizer = tokenizer
        self.processor = processor

        self.raw_data = torch.load(f"{path}/{data_files}", map_location=torch.device('cpu'))
        if size:
            self.raw_data = self.raw_data.select(range(int(size)))
        self.template = get_template_class(template)

    def get_collator(self) -> Callable[[list[dict[str, torch.Tensor]]], dict[str, torch.Tensor]]:
        return SupervisedCollator(self.tokenizer.pad_token_id)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Get a tokenized data sample by index."""
        raw_sample = self.raw_data[index]
        return raw_sample

    def __len__(self) -> int:
        """Get the number of samples in the dataset."""
        return len(self.raw_data)
    
class SupervisedCollator(DataCollatorForSeq2Seq):

    def __init__(self, tokenizer: PreTrainedTokenizer, processor: Optional["ProcessorMixin"] = None) -> None:
        """Initialize a collator."""
        super().__init__(tokenizer)
        self.processor = processor

    def __call__(self, samples: list[SupervisedSample]) -> SupervisedBatch:
        current_device = get_current_device()
        features: Dict[str, "torch.Tensor"] = super().__call__(samples)
        for k, v in features.items():
            features[k] = v.to(current_device)
        return features
