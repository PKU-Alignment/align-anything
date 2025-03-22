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
from tqdm import tqdm
from transformers.tokenization_utils import PaddingStrategy, TruncationStrategy

from align_anything.utils.multi_process import get_current_device, is_main_process
from align_anything.utils.tools import left_padding, right_padding
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


class PreferenceBatch(TypedDict, total=True):
    input_ids: torch.LongTensor  # size = (B, L)
    labels: torch.LongTensor  # size = (B, L)
    attention_mask: torch.BoolTensor  # size = (B, L)


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
        )
        if size:
            size = min(size, len(self.raw_data))
            self.raw_data = self.raw_data.select(range(int(size)))
        self.valid_indices = self.filter_indices()

    def filter_indices(self):
        valid_indices = []
        for i, item in tqdm(
            enumerate(self.raw_data),
            disable=not is_main_process(),
            total=len(self.raw_data),
            desc='Filtering valid indices',
        ):
            if not hasattr(self.template, 'check_equal'):
                valid_indices.append(i)
                continue
            if not self.template.check_equal(item):
                if hasattr(self.template, 'check_validation'):
                    if not self.template.check_validation(item):
                        continue
                valid_indices.append(i)
        return valid_indices

    def preprocess(self, raw_sample: dict[str, Any]) -> PreferenceSample:
        better_conversation, worse_conversation, meta_info = self.template.format_preference_sample(
            raw_sample
        )
        return_dict = {}
        return_dict['better_response_lens'] = len(
            self.tokenize(meta_info['better_response'], add_special_tokens=False)['input_ids'][0]
        )
        return_dict['worse_response_lens'] = len(
            self.tokenize(meta_info['worse_response'], add_special_tokens=False)['input_ids'][0]
        )
        return_dict['better_conversation'] = better_conversation
        return_dict['worse_conversation'] = worse_conversation

        return return_dict

    def tokenize(
        self,
        conversation: str,
        add_special_tokens: bool = False,
        padding: bool | str | PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        truncation: bool | str | TruncationStrategy = TruncationStrategy.LONGEST_FIRST,
        max_length: int | None = None,
    ) -> torch.LongTensor:  # size = (L,)
        """Tokenize a text string into a tensor representation."""
        if max_length is None:
            max_length = self.tokenizer.model_max_length

        return self.tokenizer(
            text=conversation,
            add_special_tokens=add_special_tokens,
            padding=padding,
            max_length=max_length,
            truncation=truncation,
            return_tensors='pt',
        )

    def get_collator(self) -> Callable[[list[dict[str, torch.Tensor]]], dict[str, torch.Tensor]]:
        return PreferenceCollator(self.tokenizer, self.tokenizer.padding_side)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Get a tokenized data sample by index."""
        raw_sample = self.raw_data[self.valid_indices[index]]
        data = self.preprocess(raw_sample)
        return data

    def __len__(self) -> int:
        """Get the number of samples in the dataset."""
        return len(self.valid_indices)


class PreferenceCollator:

    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        padding_side: str = 'right',
    ) -> None:
        """Initialize a collator."""
        self.padding_func = right_padding if padding_side == 'right' else left_padding
        self.tokenizer = tokenizer
        self.padding_side = padding_side

    def __call__(self, samples: list[PreferenceSample]) -> tuple[PreferenceBatch]:
        return_dict = {'meta_info': {}}
        current_device = get_current_device()
        concated_text = [sample['better_conversation'] for sample in samples] + [
            sample['worse_conversation'] for sample in samples
        ]  # size = (2 * B, L)
        tokenized_input = self.tokenizer(
            text=concated_text,
            return_tensors='pt',
            padding=True,
            padding_side=self.padding_side,
            return_attention_mask=True,
            add_special_tokens=False,
        )
        for key, value in tokenized_input.items():
            if isinstance(value, torch.Tensor):
                return_dict[key] = value.to(current_device)

        better_response_lens = [sample['better_response_lens'] for sample in samples]
        worse_response_lens = [sample['worse_response_lens'] for sample in samples]
        return_dict['meta_info']['response_lens'] = better_response_lens + worse_response_lens

        return return_dict
