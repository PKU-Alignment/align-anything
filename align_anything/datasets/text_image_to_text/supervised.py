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
from PIL import Image

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
        formatted_sample = self.template.format_supervised_sample(raw_sample)
        raw_text = ''
        if isinstance(formatted_sample['text'], list):
            raw_text = self.tokenizer.eos_token.join(formatted_sample['text'])
        elif isinstance(formatted_sample['text'], str):
            raw_text = formatted_sample['text'] + self.tokenizer.eos_token
        else:
            raise NotImplementedError
        
        inputs = self.tokenize(
            raw_text,
            formatted_sample['image']
        )
        inputs['input_ids'] = inputs['input_ids'][0].clone()
        formatted_prompt = formatted_sample['prompt']
        labels = inputs['input_ids'].clone()
        # mask non-assistant input
        labels[: len(self.tokenize(formatted_prompt, formatted_sample['image'])['input_ids'][0])] = IGNORE_INDEX
        inputs['labels'] = labels

        return inputs

    def get_collator(self) -> Callable[[list[dict[str, torch.Tensor]]], dict[str, torch.Tensor]]:
        return SupervisedCollator(self.tokenizer.pad_token_id)

    def tokenize(
        self,
        text: str,
        image: Image.Image,
        add_special_tokens: bool = True,
        padding: bool | str | PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        truncation: bool | str | TruncationStrategy = TruncationStrategy.LONGEST_FIRST,
        max_length: int | None = None,
    ) -> torch.LongTensor:  # size = (L,)
        """Tokenize a text string into a tensor representation."""
        if max_length is None:
            max_length = self.tokenizer.model_max_length

        return self.processor(
            image,
            text,
            return_tensors="pt",
            add_special_tokens=add_special_tokens,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
        )

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

        return_dict['pixel_values'] = torch.stack(
            [sample['pixel_values'] for sample in samples]
        ).to(current_device).squeeze(1)

        if 'aspect_ratio_ids' in samples[0].keys():
            return_dict['aspect_ratio_ids'] = torch.stack(
                [sample['aspect_ratio_ids'] for sample in samples]
            ).to(current_device).squeeze(1)

        if 'aspect_ratio_mask' in samples[0].keys():
            return_dict['aspect_ratio_mask'] = torch.stack(
                [sample['aspect_ratio_mask'] for sample in samples]
            ).to(current_device).squeeze(1)

        if 'cross_attention_mask' in samples[0].keys():
            cross_attention_mask = []
            for sample in samples:
                if 'cross_attention_mask' in sample.keys():
                    cross_attention_mask.append(sample['cross_attention_mask'].squeeze(0))
            
            return_dict['cross_attention_mask'] = right_padding(
                cross_attention_mask,
                padding_value=0,
            ).to(current_device)

        return return_dict
