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

# ref: https://github.com/OpenSafetyLab/SALAD-BENCH/blob/main/saladbench/model_utils.py

import torch


def auto_or_half_dtype():
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 8:
        return torch.float16
    else:
        return 'auto'


def auto_or_half():
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 8:
        return 'float16'
    else:
        return 'auto'


def is_ampere_or_newer():
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
