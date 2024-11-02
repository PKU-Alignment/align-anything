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
from transformers.tokenization_utils import PaddingStrategy

from align_anything.utils.multi_process import get_current_device, is_main_process
from align_anything.utils.template_registry import get_template_class
from align_anything.utils.tools import right_padding, left_padding
from datasets import load_dataset


__all__ = [
    'PreferenceDataset',
    'RightPaddingPreferenceCollator',
    'LeftPaddingPreferenceCollator',
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
        formatted_sample = self.template.format_preference_sample(raw_sample)
        return_dict = {}

        raw_better_text = formatted_sample['better_conversation']
        raw_worse_text = formatted_sample['worse_conversation']
        raw_better_response = formatted_sample['better_conversation'][-1]['content']
        raw_worse_response = formatted_sample['worse_conversation'][-1]['content']

        better_text = self.processor.apply_chat_template(raw_better_text, add_generation_prompt=False, tokenize=False)
        worse_text = self.processor.apply_chat_template(raw_worse_text, add_generation_prompt=False, tokenize=False)
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
        better_inputs = self.tokenize(text=better_text, audios=audios, padding=True)
        worse_inputs = self.tokenize(text=worse_text, audios=audios, padding=True)
        better_input_wo_padding = self.tokenize(text=better_text, audios=audios, padding=PaddingStrategy.DO_NOT_PAD)
        worse_input_wo_padding = self.tokenize(text=worse_text, audios=audios, padding=PaddingStrategy.DO_NOT_PAD)

        return_dict['better_input_ids'] = better_input_wo_padding['input_ids'][0]
        return_dict['worse_input_ids'] = worse_input_wo_padding['input_ids'][0]
        return_dict['better_response_lens'] = len(self.tokenize(raw_better_response, padding=PaddingStrategy.DO_NOT_PAD)['input_ids'][0]) + 2 # for the eos token
        return_dict['worse_response_lens'] = len(self.tokenize(raw_worse_response, padding=PaddingStrategy.DO_NOT_PAD)['input_ids'][0]) + 2 # for the eos token

        return_dict['better_feature_attention_mask'] = better_inputs['feature_attention_mask']
        return_dict['better_input_features'] = better_inputs['input_features']

        return_dict['worse_feature_attention_mask'] = worse_inputs['feature_attention_mask']
        return_dict['worse_input_features'] = worse_inputs['input_features']

        return return_dict

    def get_collator(self) -> Callable[[list[dict[str, torch.Tensor]]], dict[str, torch.Tensor]]:
        if self.processor.tokenizer.padding_side == 'left':
            return LeftPaddingPreferenceCollator(self.processor.tokenizer.pad_token_id)
        else:
            return RightPaddingPreferenceCollator(self.processor.tokenizer.pad_token_id)

    def tokenize(
        self,
        text: str,
        audios: list[torch.Tensor] | None = None,
        add_special_tokens: bool = True,
        padding: bool | str | PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
    ) -> torch.LongTensor:  # size = (L,)
        """Tokenize a text string into a tensor representation."""

        return self.processor(
            text=text, 
            audios=audios, 
            return_tensors="pt",
            padding=padding,
            sampling_rate=self.processor.feature_extractor.sampling_rate,
            add_special_tokens=add_special_tokens,
        )

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Get a tokenized data sample by index."""
        raw_sample = self.raw_data[self.valid_indices[index]]
        data = self.preprocess(raw_sample)
        return data

    def __len__(self) -> int:
        """Get the number of samples in the dataset."""
        return len(self.valid_indices)


class RightPaddingPreferenceCollator:

    def __init__(self, pad_token_id: int) -> None:
        """Initialize a collator."""
        self.pad_token_id = pad_token_id
        self.padding_func = right_padding

    def __call__(self, samples: list[PreferenceSample]) -> tuple[PreferenceBatch]:
        return_dict = {}
        current_device = get_current_device()
        input_ids = [sample['better_input_ids'] for sample in samples] + [
            sample['worse_input_ids'] for sample in samples
        ]  # size = (2 * B, L)
        input_features = [sample['better_input_features'] for sample in samples] + [
            sample['worse_input_features'] for sample in samples
        ]
        input_attention_mask = [sample['better_feature_attention_mask'] for sample in samples] + [
            sample['worse_feature_attention_mask'] for sample in samples
        ]
        return_dict['better_response_lens'] = [sample['better_response_lens'] for sample in samples]
        return_dict['worse_response_lens'] = [sample['worse_response_lens'] for sample in samples]
        return_dict['response_lens'] = return_dict['better_response_lens'] + return_dict['worse_response_lens']
        return_dict['input_ids'] = self.padding_func(input_ids, padding_value=self.pad_token_id).to(
            current_device
        )  # size = (2 * B, L)
        attention_mask = [
            input_id.new_ones(input_id.size(), dtype=torch.bool) for input_id in input_ids
        ]  # size = (2 * B, L)
        return_dict['attention_mask'] = self.padding_func(attention_mask, padding_value=0).to(
            current_device
        )  # size = (2 * B, L)

        return_dict['input_features'] = torch.cat(input_features, dim=0).to(current_device)
        return_dict['feature_attention_mask'] = torch.cat(input_attention_mask, dim=0).to(current_device)

        return return_dict

class LeftPaddingPreferenceCollator(RightPaddingPreferenceCollator):
    def __init__(self, pad_token_id: int) -> None:
        super().__init__(pad_token_id)
        self.padding_func = left_padding
