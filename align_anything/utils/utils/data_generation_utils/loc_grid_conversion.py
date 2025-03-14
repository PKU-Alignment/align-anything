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
import math

import numpy as np


def locs2grids(locations, grid_spacing):
    min_r = math.floor(min(locations, key=lambda x: x['x'])['x'] / grid_spacing)
    max_r = math.ceil(max(locations, key=lambda x: x['x'])['x'] / grid_spacing)
    min_c = math.floor(min(locations, key=lambda x: x['z'])['z'] / grid_spacing)
    max_c = math.ceil(max(locations, key=lambda x: x['z'])['z'] / grid_spacing)

    imsize = (max_r - min_r + 1, max_c - min_c + 1)

    rows = [round(loc['x'] / grid_spacing) - min_r for loc in locations]
    cols = [round(loc['z'] / grid_spacing) - min_c for loc in locations]

    valid_grid = np.zeros(imsize, dtype=bool)
    valid_grid[rows, cols] = True

    locs_grid = np.zeros(imsize + (3,), dtype=np.float32)
    locs_grid[rows, cols] = [[loc['x'], loc['y'], loc['z']] for loc in locations]

    return valid_grid, locs_grid


def grids2locs(valid_grid, locs_grid):
    locs = locs_grid[np.nonzero(valid_grid)]
    return [dict(x=loc[0], y=loc[1], z=loc[2]) for loc in locs]
