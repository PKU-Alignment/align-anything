# Copyright 2025 PKU-Alignment Team. All Rights Reserved.
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
# =============================================================================
import argparse
import os
import subprocess
import sys
from enum import Enum, unique
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

from .version import check_version_compatibility, get_version_info, get_version_string


@unique
class Command(str, Enum):
    EVAL = 'eval'
    VER = 'version'
    HELP = 'help'
    CLEAN = 'clean'
    ENV = 'env'


custom_theme = Theme(
    {
        'info': 'cyan',
        'warning': 'yellow',
        'error': 'red bold',
        'version': 'green bold',
        'title': 'blue bold',
        'subtitle': 'cyan italic',
    }
)

console = Console(theme=custom_theme)


def create_welcome_panel() -> Panel:
    info = get_version_info()
    version_text = Text()
    version_text.append('Welcome to ', style='title')
    version_text.append('Eval-Anything\n', style='title bold')
    version_text.append(f'Version: {get_version_string()}\n', style='version')
    version_text.append(f"Author: {info['author']}\n", style='info')
    version_text.append(f"License: {info['license']}", style='subtitle')

    return Panel(
        version_text,
        border_style='blue',
        title='[bold blue]Eval-Anything CLI',
        subtitle='[italic cyan]Evaluation Framework for Language Models',
    )


def create_usage_table() -> Table:
    table = Table(
        show_header=True,
        header_style='bold magenta',
        border_style='blue',
        title='[bold]Available Commands',
        width=90,
        padding=(0, 1),
    )

    table.add_column('Command', style='cyan', justify='left', width=10)
    table.add_column('Description', style='green', justify='left', width=20)
    table.add_column('Usage', style='yellow', justify='left', width=60)

    table.add_row('eval', 'Evaluate models', 'eval-anything-cli eval <config_file>')
    table.add_row('clean', 'Clean cache', 'eval-anything-cli clean <optional:cache_dir>')
    table.add_row('version', 'Show version', 'eval-anything-cli version')
    table.add_row('help', 'Show help', 'eval-anything-cli help')

    return table


def show_welcome():
    console.print(create_welcome_panel())


def show_usage():
    console.print(create_usage_table())


def show_error(message: str):
    console.print(f'[error]Error: {message}')


def show_warning(message: str):
    console.print(f'[warning]Warning: {message}')


def show_info(message: str):
    console.print(f'[info]{message}')


def run_eval(config_path: Optional[str] = None):
    """Run evaluation with specified config file"""
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Change to eval_anything directory
    eval_dir = os.path.join(script_dir, 'eval_anything')
    os.chdir(eval_dir)

    # Construct command
    cmd = ['python', '__main__.py']
    if config_path:
        cmd.extend(['--eval_info', config_path])

    # Run evaluation
    process = subprocess.run(cmd)
    sys.exit(process.returncode)


def clean_cache(cache_dir: str = 'cache'):
    cache_dir = os.path.join(os.getcwd(), cache_dir)
    if os.path.exists(cache_dir):
        for file in os.listdir(cache_dir):
            os.remove(os.path.join(cache_dir, file))
        print(f'Cleaned up cache folder: {cache_dir}')
    else:
        print(f'Cache folder not found: {cache_dir}')


def main():
    if len(sys.argv) == 1:
        show_welcome()
        console.print()
        show_usage()
        return

    command = sys.argv[1]

    try:
        if command == Command.EVAL:
            parser = argparse.ArgumentParser(description='Evaluate models')
            parser.add_argument(
                'config',
                nargs='?',
                default='configs/evaluate.yaml',
                help='Path to evaluation config file',
            )
            parser.add_argument('--gpu', type=str, default='0', help='GPU device IDs to use')
            parser.add_argument('--debug', action='store_true', help='Enable debug mode')
            args = parser.parse_args(sys.argv[2:])

            show_info(f'Starting evaluation with config: {args.config}')
            if args.debug:
                show_info('Debug mode enabled')

            run_eval(args.config)

        elif command == Command.VER:
            show_welcome()

            if not check_version_compatibility('0.0.1'):
                show_warning('Current version might not be compatible with latest features')

        elif command == Command.HELP:
            show_welcome()
            console.print()
            show_usage()

        else:
            show_error(f'Unknown command: {command}')
            console.print()
            show_usage()

    except KeyboardInterrupt:
        show_warning('\nOperation interrupted by user')
        sys.exit(1)
    except Exception as e:
        show_error(str(e))
        if '--debug' in sys.argv:
            console.print_exception()
        sys.exit(1)


if __name__ == '__main__':
    main()
