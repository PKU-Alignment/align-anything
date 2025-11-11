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

"""
评估日志记录工具

TODO 还需适配
    - 增加多模态信息的预览，可以放到wandb上
"""

import csv
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table


class EvalLogger:
    def __init__(self, name, log_dir='.', level=logging.DEBUG):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.propagate = False
        self.set_rich_console_handler()
        log_file = self.create_log_file(log_dir, name)
        self.set_file_handler(log_file)
        self.console = Console()
        self.log_dir = log_dir

    # def set_rich_console_handler(self):
    #     console_handler = RichHandler()
    #     console_handler.setLevel(logging.DEBUG)
    #     console_formatter = logging.Formatter('%(message)s')
    #     console_handler.setFormatter(console_formatter)
    #     self.logger.addHandler(console_handler)

    def set_rich_console_handler(self):
        console_handler = RichHandler(rich_tracebacks=True, show_time=True)
        console_handler.setLevel(logging.DEBUG)
        self.logger.addHandler(console_handler)

    def set_file_handler(self, log_file):
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)

    def create_log_file(self, log_dir, name):
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        log_file = os.path.join(log_dir, f'{name}.log')
        return log_file

    def log(self, level, message):
        if level == 'debug':
            self.logger.debug(message)
        elif level == 'info':
            self.logger.info(message)
        elif level == 'warning':
            self.logger.warning(message)
        elif level == 'error':
            self.logger.error(message)
        elif level == 'critical':
            self.logger.critical(message)

    def print_table(
        self,
        title: str,
        columns: List[str] = None,
        rows: List[List[Any]] = None,
        data: Dict[str, Dict[str, float]] = None,
        max_num_rows: int = None,
        to_csv: bool = False,
        csv_file_name: Optional[str] = None,
    ):
        table = Table(title=title)

        # if data and columns is None:
        #     columns = list(data.keys())

        if columns:
            for col in columns:
                table.add_column(col)

        if rows:
            if max_num_rows:
                rows = rows[:max_num_rows]
            for row in rows:
                table.add_row(*map(str, row))

        if data:
            # if max_num_rows:
            #     data = {k: v[:max_num_rows] for k, v in data.items()}
            # num_rows = len(next(iter(data.values())))
            # for i in range(num_rows):
            #     row = [data[col][i] for col in columns]
            #     table.add_row(*map(str, row))

            all_metrics = set()
            for metrics in data.values():
                all_metrics.update(metrics.keys())
            all_metrics = sorted(all_metrics)  # 确保列顺序一致
            columns = ['']
            rows = []
            table.add_column('', style='bold')
            # table.add_column("Task Name")
            for metric in all_metrics:
                columns.append(metric)
                table.add_column(metric)

            for row_label, subdict in data.items():
                row = [row_label] + [str(subdict.get(col, '')) for col in all_metrics]
                rows.append(row)
                table.add_row(*row)

        self.console.print(table)

        if to_csv:
            self.save_to_csv(columns, rows, csv_file_name)

    def save_to_csv(
        self,
        columns: List[str],
        rows: List[List[Any]],
        csv_file_name: Optional[str],
    ):
        if csv_file_name is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
            csv_file = os.path.join(self.log_dir, f'table_{timestamp}.csv')
        else:
            csv_file = os.path.join(self.log_dir, csv_file_name)

        with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(columns)
            for row in rows:
                writer.writerow(row)
