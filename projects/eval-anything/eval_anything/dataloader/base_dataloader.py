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
dataloader基类，不直接使用，而是继承后实现具体的数据加载逻辑
输入：
    - 数据集路径
    - split
    - size
    - 模态
    - 预处理方式（是否pre-tokenize）
    - 模型路径（如果需要pre-tokenize）
    - shuffle
    - num_workers
    - chat_template
    - ...
输出：
    - InferenceInput类
"""

from abc import abstractmethod
from collections import defaultdict
from typing import Dict, List

from eval_anything.utils.data_type import InferenceInput


class BaseDataLoader:

    def __init__(self, eval_cfgs, bench_cfgs, logger):
        self.eval_cfgs, self.bench_cfgs = (
            eval_cfgs,
            bench_cfgs,
        )
        self.benchmark_name = self.bench_cfgs.dataset.name
        self.num_shot = getattr(self.eval_cfgs.n_shot, self.benchmark_name, 0)
        self.enable_cot = getattr(self.eval_cfgs.cot, self.benchmark_name, False)
        self.split = self.bench_cfgs.dataset.split
        self.data_dir = self.bench_cfgs.dataset.path
        self.task_info = self.get_task_info()
        self.few_shot_data = defaultdict(list)
        self.logger = logger

    def get_task_info(self):
        if isinstance(self.bench_cfgs.task, list):
            return self.bench_cfgs.task
        else:
            tasks = [self.bench_cfgs.task]
            return tasks

    @abstractmethod
    def load_dataset(self, task_list: list[str]) -> Dict[str, List[InferenceInput]]:
        raise NotImplementedError

    @abstractmethod
    def set_fewshot_dataset(self, task: str):
        raise NotImplementedError
