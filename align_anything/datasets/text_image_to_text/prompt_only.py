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
from datasets import load_dataset


IGNORE_INDEX = -100

__all__ = [
    'PromptOnlyDataset',
    'PromptOnlyCollator',
    'PromptOnlySample',
    'PromptOnlyBatch',
]


def remove_duplicate_prompts(dict_list: list[dict[str, Any]], template):
    seen_prompts = set()
    unique_dict_list = []
    for idx in range(len(dict_list)):
        item = dict_list[idx]
        prompt = template.format_prompt_only_sample(item)[0]
        if prompt not in seen_prompts:
            unique_dict_list.append(item)
            seen_prompts.add(prompt)
    return unique_dict_list


class PromptOnlySample(TypedDict, total=True):
    input_ids: torch.LongTensor  # size = (L,)
    labels: torch.LongTensor  # size = (L,)
    pixel_values: torch.LongTensor | None  # size = (B, C, H, W)


class PromptOnlyBatch(TypedDict, total=True):
    input_ids: torch.LongTensor  # size = (B, L)
    labels: torch.LongTensor  # size = (B, L)
    attention_mask: torch.BoolTensor  # size = (B, L)
    pixel_values: torch.LongTensor | None  # size = (B, C, H, W)
    image_sizes: list[int] | None


class PromptOnlyDataset(Dataset):

    def __init__(
        self,
        path: str,
        template: str,
        tokenizer: transformers.PreTrainedTokenizer,
        processor: transformers.ProcessorMixin | transforms.Compose | None = None,
        padding_side: str = 'left',
        name: str | None = None,
        size: int | None = None,
        split: str | None = None,
        data_files: str | None = None,
        optional_args: list | str = [],
    ):
        super().__init__()
        assert path, f'You must set the valid datasets path! Here is {path}'
        assert template, f'You must set the valid template path! Here is {template}'
        self.tokenizer = tokenizer
        self.processor = processor
        raw_data_duplicated = load_dataset(
            path,
            name=name,
            split=split,
            data_files=data_files,
            *optional_args,
            trust_remote_code=True,
            verification_mode='no_checks',
        )
        self.template = template
        self.raw_data = remove_duplicate_prompts(raw_data_duplicated, self.template)
        self.padding_side = padding_side

        if size:
            self.raw_data = self.raw_data[: int(size)]

    def preprocess(self, raw_sample: dict[str, Any]) -> PromptOnlySample:
        formatted_prompt, meta_info = self.template.format_prompt_only_sample(raw_sample)
        return_dict = {}

        # return necessary information
        return_dict['image'] = meta_info['image']
        return_dict['conversation'] = formatted_prompt

        return return_dict

    def get_collator(self) -> Callable[[list[dict[str, torch.Tensor]]], dict[str, torch.Tensor]]:
        return PromptOnlyCollator(self.tokenizer.pad_token_id, self.processor, self.padding_side)

    def tokenize(
        self,
        conversation: str,
        meta_info: dict[str, Any],
        add_special_tokens: bool = True,
        padding: bool | str | PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        truncation: bool | str | TruncationStrategy = TruncationStrategy.LONGEST_FIRST,
        max_length: int | None = None,
    ) -> torch.LongTensor:  # size = (L,)
        """Tokenize a text string into a tensor representation."""
        if max_length is None:
            max_length = self.tokenizer.model_max_length

        return self.processor(
            text=conversation,
            images=meta_info['image'],
            add_special_tokens=add_special_tokens,
            padding=padding,
            max_length=max_length,
            truncation=truncation,
            return_tensors='pt',
        )

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Get a tokenized data sample by index."""
        raw_sample = self.raw_data[index]
        data = self.preprocess(raw_sample)
        return data

    def __len__(self) -> int:
        """Get the number of samples in the dataset."""
        return len(self.raw_data)


class PromptOnlyCollator:

    def __init__(
        self,
        pad_token_id: int,
        processor: transformers.ProcessorMixin | transforms.Compose | None = None,
        padding_side: str = 'left',
    ) -> None:
        """Initialize a collator."""
        self.pad_token_id = pad_token_id
        self.processor = processor
        self.padding_side = padding_side

    def __call__(self, samples: list[PromptOnlySample]) -> PromptOnlyBatch:
        return_dict = {}
        current_device = get_current_device()

        images = [sample['image'] for sample in samples]
        return_dict['meta_info']['images'] = images
        
        concated_text = [sample['conversation'] for sample in samples]

        multi_modal_padding = self.processor(
            images=images,
            text=concated_text,
            return_tensors='pt',
            padding=True,
            padding_side=self.padding_side,
        )

        for key, value in multi_modal_padding.items():
            if isinstance(value, torch.Tensor):
                return_dict[key] = value.to(current_device)

        return return_dict
