# Copyright 2024 Allen Institute for AI

# Copyright 2024-2025 Align-Anything Team. All Rights Reserved.
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
# define Python user-defined exceptions


class TaskSamplerException(Exception):
    """Task Sampler failed to find a valid sample"""

    pass


class HouseInvalidForTaskException(TaskSamplerException):
    """Task Sampler failed to find a valid sample because the house was fully impossible to generate any task for"""

    pass


class TaskSamplerInInvalidStateError(TaskSamplerException):
    """Task sampler has entered some totally invalid state from which next_task calls will definitely fail."""

    pass
