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


import math
from typing import Literal

from utils.type_utils import Vector3


def position_dist(
    p0: Vector3,
    p1: Vector3,
    ignore_y: bool = False,
    dist_fn: Literal['l1', 'l2'] = 'l2',
) -> float:
    """Distance between two points of the form {"x": x, "y": y, "z": z}."""
    if dist_fn == 'l1':
        return (
            abs(p0['x'] - p1['x'])
            + (0 if ignore_y else abs(p0['y'] - p1['y']))
            + abs(p0['z'] - p1['z'])
        )
    elif dist_fn == 'l2':
        return math.sqrt(
            (p0['x'] - p1['x']) ** 2
            + (0 if ignore_y else (p0['y'] - p1['y']) ** 2)
            + (p0['z'] - p1['z']) ** 2
        )
    else:
        raise NotImplementedError('dist_fn must be in {"l1", "l2"}.' f' You gave {dist_fn}')


def sum_dist_path(path):
    total_dist = 0
    for i in range(len(path) - 1):
        total_dist += position_dist(path[i], path[i + 1])
    return total_dist
