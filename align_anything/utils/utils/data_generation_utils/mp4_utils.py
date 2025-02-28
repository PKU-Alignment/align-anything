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
import os
from typing import Sequence, Optional, Dict, Any

import imageio
import numpy as np


def save_frames_to_mp4(
    frames: Sequence[np.ndarray],
    file_path: str,
    fps: float,
    extra_kwargs: Optional[Dict[str, Any]] = None,
):
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)

    if not isinstance(frames, np.ndarray):
        frames = np.array(frames)

    kwargs = {
        "fps": fps,
        "quality": 5,
        **(extra_kwargs if extra_kwargs is not None else {}),
    }
    imageio.mimwrite(file_path, frames, macro_block_size=1, **kwargs)
