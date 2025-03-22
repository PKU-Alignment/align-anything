# Copyright 2024 Allen Institute for AI

# Copyright 2025 Align-Anything Team. All Rights Reserved.
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


import json
from typing import Optional

import numpy as np

from align_anything.utils.spoc_utils.task_spec_to_instruction import REGISTERED_INSTRUCTION_TYPES
from align_anything.utils.spoc_utils.task_type_mapping_utils import map_task_type
from align_anything.utils.spoc_utils.type_utils import REGISTERED_TASK_PARAMS


def convert_string_to_byte(str_to_encode, max_len):
    return np.array([str_to_encode], dtype=f'S{max_len}').view('uint8')


def convert_byte_to_string(bytes_to_decode: np.ndarray, max_len: Optional[int] = None):
    if max_len is None:
        max_len = bytes_to_decode.shape[-1]
    return (bytes_to_decode.view(f'S{max_len}')[0]).decode()


def json_templated_task_string(task_info):
    task_type = task_info['task_type']

    if task_type not in REGISTERED_TASK_PARAMS:
        return 'Invalid task type.'

    keys_to_log = REGISTERED_TASK_PARAMS[task_type]
    task_info_subset = {key: task_info[key] for key in keys_to_log}

    extra_keys = [
        'task_type',
        'extras',
    ]  # nb the rest of the information is handled by other sensors
    for key in extra_keys:
        task_info_subset[key] = task_info[key]
    return json.dumps(task_info_subset)


def json_templated_spec_to_dict(task_string):
    task_dict = json.loads(task_string)
    task_dict['task_type'] = map_task_type(task_dict['task_type'])
    return task_dict


def get_natural_language_spec(task_type, task_data):
    return REGISTERED_INSTRUCTION_TYPES[map_task_type(task_type)](task_data)


def json_templated_to_NL_spec(task_string):
    task_dict = json_templated_spec_to_dict(task_string)
    return REGISTERED_INSTRUCTION_TYPES[map_task_type(task_dict['task_type'])](task_dict)


def strings_exist_in_dict_or_list(data, target_strings):
    if target_strings is None:
        return False
    if isinstance(data, dict):
        return any(strings_exist_in_dict_or_list(val, target_strings) for val in data.values())
    elif isinstance(data, list):
        return any(strings_exist_in_dict_or_list(item, target_strings) for item in data)
    elif isinstance(data, str):
        if data is not None:
            return any(target_string in data for target_string in target_strings)
    return False
