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

import librosa
import torch
import transformers
from torch.utils.data import Dataset
from torchvision import transforms
from transformers.tokenization_utils import PaddingStrategy, TruncationStrategy

from align_anything.utils.multi_process import get_current_device
from align_anything.utils.template_registry import get_template_class
from align_anything.utils.tools import right_padding
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
        subset: str | None = None,
        data_files: str | None = None,
        optional_args: list | str = [],
    ):
        super().__init__()
        assert path, f'You must set the valid datasets path! Here is {path}'
        assert template, f'You must set the valid template path! Here is {template}'
        self.tokenizer = tokenizer
        self.processor = processor
        self.template = get_template_class(template)

        if isinstance(optional_args, str):
            optional_args = [optional_args]
        self.raw_data = load_dataset(
            path,
            name=name,
            split=split,
            data_files=data_files,
            subset=subset,
            *optional_args,
            trust_remote_code=True,
        )
        self.valid_indices = self.filter_indices()

        if size:
            size = min(size, len(self.raw_data))
            self.raw_data = self.raw_data.select(range(int(size)))

    def filter_indices(self):
        valid_indices = []
        for i, item in enumerate(self.raw_data):
            if not self.template.check_equal(item):
                valid_indices.append(i)
        return valid_indices

    def preprocess(self, raw_sample: dict[str, Any]) -> PreferenceSample:
        formatted_sample = self.template.format_sample(raw_sample)
        return_dict = {}

        raw_better_text = formatted_sample['better_conversation']
        raw_worse_text = formatted_sample['worse_conversation']

        better_text = self.processor.apply_chat_template(raw_better_text, add_generation_prompt=True, tokenize=False)
        worse_text = self.processor.apply_chat_template(raw_worse_text, add_generation_prompt=True, tokenize=False)
        audios = []

        for message in raw_better_text:
            if isinstance(message["content"], list):
                for ele in message["content"]:
                    if ele["type"] == "audio":
                        audios.append(
                            librosa.load(
                                ele['audio_url'], 
                                sr=self.processor.feature_extractor.sampling_rate)[0]
                        )
        better_inputs = self.tokenize(text=better_text, audios=audios)
        worse_inputs = self.tokenize(text=worse_text, audios=audios)
        return_dict['better_input_ids'] = better_inputs['input_ids'][0]
        return_dict['worse_input_ids'] = worse_inputs['input_ids'][0]
        return_dict['feature_attention_mask'] = better_inputs['feature_attention_mask']

        return return_dict

    def get_collator(self) -> Callable[[list[dict[str, torch.Tensor]]], dict[str, torch.Tensor]]:
        return PreferenceCollator(self.tokenizer.pad_token_id)

    def tokenize(
        self,
        text: str,
        audios: list,
        add_special_tokens: bool = True,
        padding: bool | str | PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        truncation: bool | str | TruncationStrategy = TruncationStrategy.LONGEST_FIRST,
        max_length: int | None = None,
    ) -> torch.LongTensor:  # size = (L,)
        """Tokenize a text string into a tensor representation."""
        if max_length is None:
            max_length = self.tokenizer.model_max_length

        return self.processor(
            text=text, 
            audios=audios, 
            return_tensors="pt",
            add_special_tokens=add_special_tokens,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            sampling_rate=self.processor.feature_extractor.sampling_rate,
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

    def __init__(self, pad_token_id: int) -> None:
        """Initialize a collator."""
        self.pad_token_id = pad_token_id

    def __call__(self, samples: list[PreferenceSample]) -> tuple[PreferenceBatch]:
        return_dict = {}
        current_device = get_current_device()
        input_ids = [sample['better_input_ids'] for sample in samples] + [
            sample['worse_input_ids'] for sample in samples
        ]  # size = (2 * B, L)
        feature_attention_mask = [sample['feature_attention_mask'] for sample in samples]

        return_dict['input_ids'] = right_padding(input_ids, padding_value=self.pad_token_id).to(
            current_device
        )  # size = (2 * B, L)
        attention_mask = [
            input_id.new_ones(input_id.size(), dtype=torch.bool) for input_id in input_ids
        ]  # size = (2 * B, L)
        return_dict['attention_mask'] = right_padding(attention_mask, padding_value=0).to(
            current_device
        )  # size = (2 * B, L)

        return_dict['feature_attention_mask'] = right_padding(feature_attention_mask, padding_value=0).to(
            current_device
        )  # size = (2 * B, L)

        return return_dict
