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
from align_anything.utils.tools import right_padding, left_padding
from datasets import load_dataset


__all__ = [
    'PreferenceDataset',
    'PreferenceCollator',
    'PreferenceSample',
    'PreferenceBatch',
]


class PreferenceSample(TypedDict, total=True):
    input_ids: torch.LongTensor  # size = (L,)
    labels: torch.LongTensor  # size = (L,)
    pixel_values: torch.LongTensor | None  # size = (B, C, H, W)


class PreferenceBatch(TypedDict, total=True):
    input_ids: torch.LongTensor  # size = (B, L)
    labels: torch.LongTensor  # size = (B, L)
    attention_mask: torch.BoolTensor  # size = (B, L)
    pixel_values: torch.LongTensor | None  # size = (B, C, H, W)


class PreferenceDataset(Dataset):

    def __init__(
        self,
        path: str,
        template: str,
        tokenizer: transformers.PreTrainedTokenizer,
        processor: transformers.ProcessorMixin | transforms.Compose | None = None,
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
        self.template = template

        if isinstance(optional_args, str):
            optional_args = [optional_args]
        self.raw_data = load_dataset(
            path,
            name=name,
            split=split,
            data_files=data_files,
            *optional_args,
            trust_remote_code=True,
            verification_mode="no_checks"
        )
        self.valid_indices = self.filter_indices()

        if size:
            size = min(size, len(self.raw_data))
            self.raw_data = self.raw_data.select(range(int(size)))
        
    def filter_indices(self):
        valid_indices = []
        for i, item in enumerate(self.raw_data):
            if not self.template.check_equal(item):
                if hasattr(self.template, 'check_validation'):
                    if not self.template.check_validation(item):
                        continue
                valid_indices.append(i)
        return valid_indices

    def preprocess(self, raw_sample: dict[str, Any]) -> PreferenceSample:
        better_conversation, worse_conversation, meta_info = self.template.format_preference_sample(raw_sample)
        return_dict = {}
        return_dict['better_response_lens'] = len(self.tokenize(meta_info['better_response'], meta_info, add_special_tokens=False)['input_ids'][0])
        return_dict['worse_response_lens'] = len(self.tokenize(meta_info['worse_response'], meta_info, add_special_tokens=False)['input_ids'][0])
        return_dict['better_conversation'] = better_conversation
        return_dict['worse_conversation'] = worse_conversation
        return_dict['image'] = meta_info['image']

        return return_dict

    def get_collator(self) -> Callable[[list[dict[str, torch.Tensor]]], dict[str, torch.Tensor]]:
        return PreferenceCollator(self.tokenizer.pad_token_id, self.processor, self.tokenizer.padding_side)

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
        raw_sample = self.raw_data[self.valid_indices[index]]
        data = self.preprocess(raw_sample)
        return data

    def __len__(self) -> int:
        """Get the number of samples in the dataset."""
        return len(self.valid_indices)


class PreferenceCollator:

    def __init__(self, pad_token_id: int, processor: transformers.ProcessorMixin | transforms.Compose | None = None, padding_side: str = 'right') -> None:
        """Initialize a collator."""
        self.pad_token_id = pad_token_id
        self.padding_func = right_padding
        self.processor = processor
        self.padding_side = padding_side

    def __call__(self, samples: list[PreferenceSample]) -> tuple[PreferenceBatch]:
        return_dict = {}
        current_device = get_current_device()

        images = [sample['image'] for sample in samples] * 2
        return_dict['images'] = images
        concated_text = [sample['better_conversation'] for sample in samples] + [sample['worse_conversation'] for sample in samples]

        multi_modal_padding = self.processor(images=images, text=concated_text, return_tensors='pt', padding=True, padding_side=self.padding_side)

        for key, value in multi_modal_padding.items():
            if isinstance(value, torch.Tensor):
                return_dict[key] = value.to(current_device)

        attention_mask = [
            input_id.new_ones(input_id.size(), dtype=torch.bool) for input_id in return_dict['input_ids']
        ]  # size = (2 * B, L)
        return_dict['attention_mask'] = self.padding_func(attention_mask, padding_value=0).to(
            current_device
        )  # size = (2 * B, L)
        return_dict['better_response_lens'] = [sample['better_response_lens'] for sample in samples]
        return_dict['worse_response_lens'] = [sample['worse_response_lens'] for sample in samples]
        return_dict['response_lens'] = return_dict['better_response_lens'] + return_dict['worse_response_lens']
        return return_dict
    