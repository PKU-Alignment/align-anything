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
"""Interactive chatbot."""

from __future__ import annotations

import abc
import dataclasses
import os
import re
from enum import Enum
from threading import Thread
from typing import Generator, Iterable, Iterator, NoReturn, overload

import requests
import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    GenerationConfig,
    LlavaForConditionalGeneration,
    TextIteratorStreamer,
)

from align_anything.models.pretrained_model import load_pretrained_models
from align_anything.utils.multi_process import to_device
from align_anything.utils.template_registry import get_template_class


__all__ = [
    'CODE_BLOCK_PATTERN',
    'ModelArgs',
    'Chatbot',
    'ChatbotList',
    'EndOfDialogue',
    'SpecialCommand',
]


# pylint: disable=line-too-long
CODE_BLOCK_PATTERN = re.compile(
    r"""
    ^                                               # the code block must start at the beginning of a line
    (?P<prefix>
        (?:[ ]*)                                    # leading spaces
        (?P<fence>                                  # the code block must start with "```" or "~~~"
            (?P<char>(?P<backtick>`)|(?P<tilde>~))
            (?P=char){2}
            (?P=char)*
        )
        (?:[ ]*)                                    # optional spaces
        (?P<language>[\w\-\+\./#]+)?                # optional language identifier
        (?:[ ]*)                                    # optional spaces
        \n                                          # the prefix must end with a newline
    )
    (?:
        (?P<content>.*?)                            # the code block content searched in non-greedy mode
        \n                                          # the non-empty code block must end with a newline
    )?                                              # or an empty code block
    (?P<suffix>
        (?:[ ]*)                                    # leading spaces
        (?P=fence)                                  # the code block must end with "```" or "~~~" same as the prefix
        (?:[ ]*)                                    # optional spaces
    )
    $                                               # the code block must end with a newline
    """,
    flags=re.DOTALL | re.MULTILINE | re.VERBOSE,
)
# pylint: enable=line-too-long


class SpecialCommand(Enum):
    """Special commands for the chatbot."""

    RESET = '/reset: Reset the dialogue context.'
    CLEAR = '/clear: Clear the dialogue history and clear screen.'
    QUIT = '/quit: End the dialogue and quit.'
    EXIT = '/exit: End the dialogue and quit.'
    REGENERATE = '/regenerate: Regenerate the last response.'
    HELP = '/help: Show this help message.'
    IMAGE = '/image: Select an image as input.'
    VLM = '/vlm: Whether to use VLM model'

    def __eq__(self, other: object) -> bool:
        """Test if the command is equal to the given string."""
        if isinstance(other, str):
            return self.command == other
        return super().__eq__(other)

    def __hash__(self) -> int:
        """Hash the command."""
        return hash(self.command)

    @property
    def command(self) -> str:
        """Get the command string."""
        return self.value.partition(':')[0]

    @property
    def help(self) -> str:
        """Get the help message."""
        return self.value.partition(':')[-1].strip()

    @classmethod
    @property
    def commands(cls) -> list[str]:
        """Get all the commands."""
        return [command.command for command in cls]


class EndOfDialogue(Exception):
    """Exception raised when the dialogue ends."""


class AbstractChatbot:
    """Abstract chatbot."""

    @property
    @abc.abstractmethod
    def round(self) -> int:
        """Get the step counter of the current dialogue."""

    @property
    def help_message(self) -> str:
        """Get the help message."""
        max_command_length = max(len(command.command) for command in SpecialCommand)
        return '\n'.join(
            ['A help message for the chatbot. List of all the special commands:']
            + [
                f'  {command.command:<{max_command_length}} - {command.help}'
                for command in SpecialCommand
            ],
        )

    @abc.abstractmethod
    def __call__(self, text: str, stream: bool = False) -> Iterable[str] | Iterable[Iterable[str]]:
        """Generate the response to the given text."""

    @abc.abstractmethod
    def generator(
        self,
        text: str,
        stream: bool = False,
    ) -> Generator[str, None, None] | Iterable[Generator[str, None, None]]:
        """Generate the response to the given text."""

    @abc.abstractmethod
    def regenerator(
        self,
        stream: bool = False,
    ) -> Generator[str, None, None] | Iterable[Generator[str, None, None]]:
        """Regenerate the last response."""

    @abc.abstractmethod
    def reset(self) -> None:
        """Reset the dialogue context."""

    def clear(self) -> None:
        """Clear the dialogue history."""
        self.reset()

    def exit(self) -> NoReturn:
        """Exit the dialogue."""
        raise EndOfDialogue

    def quit(self) -> NoReturn:
        """Exit the dialogue."""
        raise EndOfDialogue


@dataclasses.dataclass
class ModelArgs:
    model_name_or_path: str | os.PathLike
    temperature: float = 1.0
    max_length: int = 4096
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    dtype: torch.dtype | str | None = 'auto'
    template: str = 'Dialogue'
    vlm: str = 'False'


class Chatbot(AbstractChatbot):
    """Interactive chatbot."""

    def __init__(
        self,
        model_name_or_path: str | os.PathLike,
        temperature: float = 1.0,
        max_length: int = 4096,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        dtype: torch.dtype | str | None = 'auto',
        template: str = 'Dialogue',
        image_source: str = '',
        vlm: str = 'False',
    ) -> None:
        """Initialize the chatbot."""

        self.name = os.path.basename(os.path.normpath(model_name_or_path))
        self.template = get_template_class(template)
        if vlm == 'True':
            self.vlm = True
        else:
            self.vlm = False
        self.messages = []

        if not self.vlm:
            self.model, self.tokenizer, self.processor = load_pretrained_models(
                model_name_or_path,
                model_max_length=max_length,
                auto_device_mapping=torch.cuda.is_available(),
                dtype=dtype,
                trust_remote_code=True,
            )
            self.max_length = 4096
            self.tokenizer.model_max_length = self.max_length
            self.generation_config = GenerationConfig(
                do_sample=(temperature > 0.0),
                temperature=temperature,
                max_new_tokens=max_length,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        else:
            self.model = LlavaForConditionalGeneration.from_pretrained(model_name_or_path)
            self.processor = AutoProcessor.from_pretrained(model_name_or_path)

        self.PROMPT_BEGIN = self.template.system_prompt
        self.dialogue = ''
        self.last_dialogue = ''
        self.last_input = ''
        self.last_response = ''
        self.inputs = []
        self.responses = []
        self.image_source = image_source

    @property
    def round(self) -> int:
        """Get the step counter of the current dialogue."""
        return len(self.inputs)

    def reset(self) -> None:
        """Reset the dialogue context."""
        self.dialogue = self.PROMPT_BEGIN
        self.last_dialogue = ''
        self.last_input = ''
        self.last_response = ''
        self.inputs.clear()
        self.responses.clear()

    def __call__(self, text: str, stream: bool = False, image_source: str = '') -> Iterable[str]:
        """Generate the response to the given text."""

        self.image_source = image_source

        return self.generator(text, stream=stream)

    def generator(self, text: str, stream: bool = False) -> Generator[str, None, None]:
        """Generate the response to the given text."""
        if self.vlm and self.image_source:
            text = '<image>\n' + text
            if self.image_source.startswith(('http://', 'https://')):
                image = Image.open(requests.get(self.image_source, stream=True).raw)
            else:
                image = Image.open(self.image_source)
        self.last_input = text
        self.last_dialogue = self.dialogue
        raw_sample = {
            'instruction': '',
            'input': text,
            'output': '',
        }
        prompt = self.template.format_sample(raw_sample)['prompt']
        self.inputs.append(text)

        input = self.dialogue + prompt

        if self.vlm and self.image_source:

            inputs = self.processor(text=input, images=image, return_tensors='pt')
            generate_ids = self.model.generate(**inputs, max_new_tokens=128)
            output = self.processor.batch_decode(
                generate_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
            )[0]
            input = '<s> ' + input
            response = output[len(input) + 1 :].strip()

            yield response
        elif self.vlm:
            inputs = self.processor(text=input, return_tensors='pt')
            generate_ids = self.model.generate(**inputs, max_new_tokens=128)
            output = self.processor.batch_decode(
                generate_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
            )[0]
            response = output.replace(input, '', 1)
            yield response
        # dialogue = self.dialogue + PROMPT_USER.format(input=text) + PROMPT_ASSISTANT
        tokenized = to_device(
            self.tokenizer(input, return_tensors='pt'),
            device=('cuda' if torch.cuda.is_available() else None),
        )
        decoded_text = self.tokenizer.decode(tokenized['input_ids'][0], skip_special_tokens=True)

        if stream:
            streamer = TextIteratorStreamer(
                tokenizer=self.tokenizer,
                skip_prompt=True,
                skip_special_tokens=True,
            )
            daemon = Thread(
                target=self.model.generate,
                kwargs={
                    'input_ids': tokenized['input_ids'],
                    'attention_mask': tokenized['attention_mask'],
                    'generation_config': self.generation_config,
                    'streamer': streamer,
                },
                daemon=True,
            )
            daemon.start()

            response = ''
            for new_token in streamer:
                response += new_token
                yield response

            daemon.join()
        else:

            output = self.model.generate(
                input_ids=tokenized['input_ids'].to(self.model.device),
                attention_mask=tokenized['attention_mask'].to(self.model.device),
                generation_config=self.generation_config,
            )

            dialogue_text = self.tokenizer.decode(output[0], skip_special_tokens=False)
            clean_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
            clean_response = clean_text.replace(decoded_text, '', 1)

            response = dialogue_text.replace(self.dialogue, '', 1)

        self.last_response = response
        self.responses.append(response)
        raw_sample = {
            'instruction': '',
            'input': text,
            'output': response,
        }
        message = self.template.format_sample(raw_sample)['text']
        self.dialogue += message + self.tokenizer.eos_token

        yield clean_response

    def regenerator(self, stream: bool = False) -> Generator[str, None, None]:
        """Regenerate the last response."""
        if len(self.inputs) == 0:
            return ['WRONG COMMAND: Empty dialogue history. No input to regenerate.']

        self.dialogue = self.last_dialogue
        self.inputs.pop()
        self.responses.pop()
        return self.generator(self.last_input, stream=stream)


class ChatbotList(AbstractChatbot):
    """List of chatbot."""

    def __init__(self, chatbots: Iterable[Chatbot | ModelArgs]) -> None:
        self.chatbots = [
            chatbot if isinstance(chatbot, Chatbot) else Chatbot(**dataclasses.asdict(chatbot))
            for chatbot in chatbots
        ]

    def __len__(self) -> int:
        return len(self.chatbots)

    def __iter__(self) -> Iterator[Chatbot]:
        return iter(self.chatbots)

    @overload
    def __getitem__(self, index: int) -> Chatbot: ...

    @overload
    def __getitem__(self, index: slice) -> ChatbotList: ...

    def __getitem__(self, index: int | slice) -> Chatbot | ChatbotList:
        if isinstance(index, slice):
            return ChatbotList(self.chatbots[index])
        return self.chatbots[index]

    @property
    def round(self) -> int:
        return max((chatbot.round for chatbot in self.chatbots), default=0)

    def reset(self) -> None:
        """Reset the dialogue context."""
        for chatbot in self.chatbots:
            chatbot.reset()

    def clear(self) -> None:
        """Clear the dialogue history."""
        for chatbot in self.chatbots:
            chatbot.clear()

    def __call__(
        self, text: str, stream: bool = False, image_source: str = ''
    ) -> Iterable[Iterable[str]]:
        """Generate the response to the given text."""
        for chatbot in self.chatbots:
            yield chatbot(text, stream=stream, image_source=image_source)

    def generator(self, text: str, stream: bool = False) -> Iterable[Generator[str, None, None]]:
        """Generate the response to the given text."""
        for chatbot in self.chatbots:
            yield chatbot.generator(text, stream=stream)

    def regenerator(self, stream: bool = False) -> Iterable[Generator[str, None, None]]:
        """Regenerate the last response."""
        for chatbot in self.chatbots:
            yield chatbot.regenerator(stream=stream)
