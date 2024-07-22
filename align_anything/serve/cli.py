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
"""Command line interface for interacting with a model."""

from __future__ import annotations

import argparse
import itertools
from typing import Generator, Iterable

import torch
from rich.console import Console
from rich.live import Live
from rich.syntax import Syntax
from rich.text import Text
from transformers.utils import is_torch_bf16_gpu_available, is_torch_tf32_available

from align_anything.serve.chatbot import (
    CODE_BLOCK_PATTERN,
    ChatbotList,
    EndOfDialogue,
    ModelArgs,
    SpecialCommand,
)
from align_anything.utils.tools import str2bool


class CLI:
    STYLES: Iterable[str] = itertools.cycle(('blue', 'red', 'cyan', 'magenta'))

    def __init__(self, *model_args: ModelArgs, stream: bool = True) -> None:
        self.model_args = model_args
        self.stream = stream
        self.console = Console(soft_wrap=False, markup=False, emoji=False, highlight=False)
        self.console.print('Loading model...', style='bold yellow')
        self.chatbots = ChatbotList(self.model_args)
        chatbot_names = [chatbot.name for chatbot in self.chatbots]
        self.chatbot_names = [
            name[:14] + '..' if len(name) > 16 else name for name in chatbot_names
        ]
        self.max_name_length = max(len(name) for name in self.chatbot_names)
        self.styles = [style for style, _ in reversed(tuple(zip(self.STYLES, self.chatbots)))]

        self.console.print('Model loaded. ', style='bold yellow', end='')
        self.do_generate = True
        self.image_source = ''
        self.clear()

    def reset(self) -> None:
        self.chatbots.reset()
        self.console.print(
            (
                'HINT: '
                'Type "Ctrl + C" or "Ctrl + D" to exit. '
                'Type "/clear" to clear dialogue history. '
                'Type "/help" to see help message.'
            ),
        )
        self.console.print()

    def clear(self) -> None:
        self.chatbots.clear()
        self.console.print(
            (
                'HINT: '
                'Type "Ctrl + C" or "Ctrl + D" to exit. '
                'Type "/clear" to clear dialogue history. '
                'Type "/help" to see help message.'
            ),
        )
        self.console.print()

    def run(self) -> None:
        try:
            while True:
                self.do_generate = True
                self.console.print(
                    f'[{self.chatbots.round + 1}] Human: ',
                    style='bold green',
                    end='',
                )
                try:
                    text = self.console.input()
                except UnicodeDecodeError as ex:
                    self.console.print('ERROR: ', style='bold red', end='')
                    self.console.print(f'Invalid input. Got UnicodeDecodeError: {ex}')
                    self.console.print('Please try again.')
                    continue

                if text == SpecialCommand.RESET:
                    self.console.print()
                    self.console.rule(characters='.')
                    self.console.print()
                    self.reset()
                    continue
                if text == SpecialCommand.CLEAR:
                    self.console.clear()
                    self.clear()
                    continue
                if text == SpecialCommand.HELP:
                    message, _, commands = self.chatbots.help_message.partition('\n')
                    self.console.print(message, style='bold yellow')
                    self.console.print(commands)
                    self.console.print()
                    continue
                if len(text.split()) and text.split()[0] == SpecialCommand.IMAGE:
                    self.do_generate = False
                    if not self.model_args[0].vlm:
                        # print something
                        self.console.print(
                            'WRONG COMMAND: /image command is only available for VLM model.Try setting --vlm True',
                            style='bold yellow',
                        )
                    else:
                        parts = text.split()[1:]
                        if len(parts) > 1:
                            self.console.print(
                                'WRONG COMMAND: Too many arguments for /image command.',
                                style='bold yellow',
                            )

                        elif len(parts) == 0:
                            self.console.print(
                                'WRONG COMMAND: No arguments for /image command.',
                                style='bold yellow',
                            )
                        else:
                            self.image_source = parts[0]
                            self.console.print(
                                f'Image source set to {self.image_source}.', style='bold yellow'
                            )

                if self.do_generate:
                    for response_generator, name, style in zip(
                        self.chatbots(
                            text=text, stream=self.stream, image_source=self.image_source
                        ),
                        self.chatbot_names,
                        self.styles,
                    ):
                        self.render(response_generator, name, style)

                self.console.print()
                if self.do_generate and self.image_source != '':
                    exit()

        except (KeyboardInterrupt, EOFError, EndOfDialogue) as ex:
            if not isinstance(ex, EndOfDialogue):
                self.console.print()
            self.console.print('Bye!', style='bold yellow')

    def render(self, response_generator: Generator[str, None, None], name: str, style: str) -> None:
        response = ''
        if self.stream:
            with Live(console=self.console, refresh_per_second=25, transient=True) as live:
                prefix = 'Generating...\n'
                for response in response_generator:
                    live.update(Text(prefix + response, style=f'dim {style}'))
        else:
            response = next(response_generator)

        self.console.print(f'[{self.chatbots.round}] Assistant (', style=f'bold {style}', end='')
        self.console.print(name, end='')
        self.console.print(
            ')'.ljust(self.max_name_length - len(name) + 1) + ' : ',
            style=f'bold {style}',
            end='',
        )

        response = response.lstrip(' ')
        if '\n' in response:
            self.console.print('⤦', style=f'dim {style}')
        while match := CODE_BLOCK_PATTERN.search(response):
            preamble, response = response[: match.start()], response[match.end() :]
            self.console.print(preamble, style=style, end='')
            self.console.print(match.group('prefix'), style=f'bold italic {style}', end='')
            if (content := match.group('content')) is not None:
                self.console.print(
                    Syntax(
                        content,
                        lexer=(match.group('language') or 'text').lower(),
                        background_color='default',
                    ),
                    end='',
                )
            self.console.print(match.group('suffix'), style=f'bold italic {style}', end='')
        self.console.print(response, style=style, soft_wrap=True)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Talking to one or more model(s) in cli mode.')
    parser.add_argument(
        '--model_name_or_path',
        type=str,
        nargs='+',
        help='Path to the model checkpoint or its name.',
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=1.0,
        help='The value used to module the next token probabilities.',
    )
    parser.add_argument(
        '--max_length',
        type=int,
        default=4096,
        help='Maximum sequence length of generation.',
    )
    parser.add_argument(
        '--top_p',
        type=float,
        default=1.0,
        help=(
            'If set to float < 1, only the smallest set of most probable tokens with '
            'probabilities that add up to`top_p` or higher are kept for generation.'
        ),
    )
    parser.add_argument(
        '--repetition_penalty',
        type=float,
        default=1.0,
        help='The parameter for repetition penalty. 1.0 means no penalty.',
    )
    parser.add_argument(
        '--stream',
        action='store_true',
        help='Whether to stream the output.',
        default=False,
    )
    parser.add_argument(
        '--fp16',
        type=str2bool,
        default=False,
        help='Whether to use float16 precision.',
    )
    parser.add_argument(
        '--bf16',
        type=str2bool,
        default=False,
        help='Whether to use bfloat16 precision.',
    )
    parser.add_argument(
        '--tf32',
        type=str2bool,
        default=None,
        help='Whether to use tf32 mix precision.',
    )
    parser.add_argument(
        '--template',
        type=str,
        default='Dialogue',
        help='Model template',
    )
    parser.add_argument(
        '--vlm',
        type=str,
        default=False,
        help='Whether to use VLM model',
    )

    args = parser.parse_args()
    if args.fp16 and args.bf16:
        parser.error('Cannot use both bf16 and fp16 precision.')
    if args.bf16 and not is_torch_bf16_gpu_available():
        parser.error(
            'bf16 precision is not supported on this GPU. '
            'Please disable `--bf16` flag or use another precision flag (e.g., `--fp16`).',
        )
    if args.tf32 is not None and is_torch_tf32_available():
        torch.backends.cuda.matmul.allow_tf32 = args.tf32

    return args


def main(args: argparse.Namespace | None = None) -> None:
    if args is None:
        args = parse_arguments()

    common_model_args = {
        'temperature': args.temperature,
        'max_length': args.max_length,
        'top_p': args.top_p,
        'repetition_penalty': args.repetition_penalty,
        'dtype': (torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else 'auto')),
        'template': args.template,
        'vlm': args.vlm,
    }
    cli = CLI(
        *(
            ModelArgs(model_name_or_path=model_name_or_path, **common_model_args)
            for model_name_or_path in args.model_name_or_path
        ),
        stream=args.stream,
    )
    cli.run()


if __name__ == '__main__':
    main()
