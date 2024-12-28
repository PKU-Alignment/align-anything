# Copyright 2024 PKU-Alignment Team and Lagent Team. All Rights Reserved.
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


LLAMA31_META = [
    dict(
        role='system',
        begin='<|start_header_id|>system<|end_header_id|>\n\n',
        end_of_turn='<|eot_id|>\n',
    ),
    dict(
        role='user',
        begin='<|start_header_id|>user<|end_header_id|>\n\n',
        end_of_turn='<|eot_id|>\n',
    ),
    dict(
        role='assistant',
        begin='<|start_header_id|>assistant<|end_header_id|>\n\n',
        end_of_turn='<|eot_id|>\n',
        end_of_message='<|eom_id|>\n',
    ),
    dict(
        role='ipython',
        begin='<|start_header_id|>ipython<|end_header_id|>\n\n',
        end_of_turn='<|eot_id|>\n',
    ),
]
