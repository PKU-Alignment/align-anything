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
"""The logger utility."""

from __future__ import annotations

import argparse
import atexit
import logging
import os
import pathlib
import sys
from typing import TYPE_CHECKING, Any, ClassVar, Literal, TextIO
from typing_extensions import Self  # Python 3.11+

import torch.utils.tensorboard as tensorboard
import wandb
import yaml
from rich.console import Console  # pylint: disable=wrong-import-order
from rich.table import Table  # pylint: disable=wrong-import-order
from tqdm import tqdm  # pylint: disable=wrong-import-order

from align_anything.utils.multi_process import is_main_process, rank_zero_only


if TYPE_CHECKING:
    import wandb.sdk.wandb_run


LoggerLevel = Literal['CRITICAL', 'FATAL', 'ERROR', 'WARN', 'WARNING', 'INFO', 'DEBUG', 'NOTSET']
_LOGGER = logging.getLogger('safe_rlhf')


def set_logger_level(level: LoggerLevel | None = None) -> None:
    """Set the logger level."""

    level = level or os.getenv('LOGLEVEL')
    if level is None:
        return

    level = level.upper()
    if is_main_process():
        print(f'Set logger level to {level}.')

    logging.basicConfig(level=level)
    _LOGGER.setLevel(level)
    logging.getLogger('DeepSpeed').setLevel(level)
    logging.getLogger('transformers').setLevel(level)
    logging.getLogger('datasets').setLevel(level)


class Logger:
    """The logger utility."""

    _instance: ClassVar[Logger] = None  # singleton pattern
    writer: tensorboard.SummaryWriter | None
    wandb: wandb.sdk.wandb_run.Run | None

    def __new__(
        cls,
        log_type: Literal['none', 'wandb', 'tensorboard'] = 'none',
        log_dir: str | os.PathLike | None = None,
        log_project: str | None = None,
        log_run_name: str | None = None,
        config: dict[str, Any] | argparse.Namespace | None = None,
    ) -> Self:
        """Create and get the logger singleton."""
        assert log_type in {
            'none',
            'wandb',
            'tensorboard',
        }, f'log_type should be one of [tensorboard, wandb, none], but got {log_type}'

        if cls._instance is None:
            self = cls._instance = super().__new__(
                cls,
            )

            self.log_type = log_type
            self.log_dir = log_dir
            self.log_project = log_project
            self.log_run_name = log_run_name
            self.writer = None
            self.wandb = None

            if is_main_process():
                if self.log_type == 'tensorboard':
                    self.writer = tensorboard.SummaryWriter(log_dir)
                elif self.log_type == 'wandb':
                    self.wandb = wandb.init(
                        project=log_project,
                        name=log_run_name,
                        dir=log_dir,
                        config=config,
                    )

                if log_dir is not None:
                    log_dir = pathlib.Path(log_dir).expanduser().absolute()

                    log_dir.mkdir(parents=True, exist_ok=True)
                    with open(log_dir / 'arguments.yaml', 'w', encoding='utf-8') as file:
                        yaml.dump(config, file, default_flow_style=False, allow_unicode=True)
                    with (log_dir / 'environ.txt').open(mode='wt', encoding='utf-8') as file:
                        file.write(
                            '\n'.join(
                                (f'{key}={value}' for key, value in sorted(os.environ.items())),
                            ),
                        )

            atexit.register(self.close)
        else:
            assert log_dir is None, 'logger has been initialized'
            assert log_project is None, 'logger has been initialized'
            assert log_run_name is None, 'logger has been initialized'
        return cls._instance

    @rank_zero_only
    def log(self, metrics: dict[str, Any], step: int) -> None:
        """Log a dictionary of scalars to the logger backend."""
        tags = {key.rpartition('/')[0] for key in metrics}
        metrics = {**{f'{tag}/step': step for tag in tags}, **metrics}
        if self.log_type == 'tensorboard':
            for key, value in metrics.items():
                self.writer.add_scalar(key, value, global_step=step)
        elif self.log_type == 'wandb':
            self.wandb.log(metrics, step=step)

    @rank_zero_only
    def close(self) -> None:
        """Close the logger backend."""
        if self.log_type == 'tensorboard':
            self.writer.close()
        elif self.log_type == 'wandb' and self.wandb:
            self.wandb.finish()

    @staticmethod
    @tqdm.external_write_mode()
    @rank_zero_only
    def print(
        *values: object,
        sep: str | None = ' ',
        end: str | None = '\n',
        file: TextIO | None = None,
        flush: bool = False,
    ) -> None:
        """Print a message in the main process."""
        print(*values, sep=sep, end=end, file=file or sys.stdout, flush=flush)

    @staticmethod
    @tqdm.external_write_mode()
    @rank_zero_only
    def print_table(
        title: str,
        columns: list[str] | None = None,
        rows: list[list[Any]] | None = None,
        data: dict[str, list[Any]] | None = None,
        max_num_rows: int | None = None,
    ) -> None:
        """Print a table in the main process."""
        if data is not None:
            if columns is not None or rows is not None:
                raise ValueError(
                    '`logger.print_table` should be called with both `columns` and `rows`, '
                    'or call with `data`.',
                )
            columns = list(data.keys())
            rows = list(zip(*data.values()))
        elif columns is None or rows is None:
            raise ValueError(
                '`logger.print_table` should be called with both `columns` and `rows`, '
                'or call with `data`.',
            )

        if max_num_rows is None:
            max_num_rows = len(rows)

        rows = [[str(item) for item in row] for row in rows]

        table = Table(title=title, show_lines=True, title_justify='left')
        for column in columns:
            table.add_column(column)
        for row in rows[:max_num_rows]:
            table.add_row(*row)
        Console(soft_wrap=True, markup=False, emoji=False).print(table)
