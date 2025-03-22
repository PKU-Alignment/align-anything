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


import os
from typing import Any, Dict, Optional

import compress_json

from align_anything.utils.spoc_utils.constants.objaverse_data_dirs import OBJAVERSE_ANNOTATIONS_PATH


_OBJAVERSE_ANNOTATIONS: Optional[Dict[str, Any]] = None


def get_objaverse_annotations():
    global _OBJAVERSE_ANNOTATIONS
    if _OBJAVERSE_ANNOTATIONS is None:
        if not os.path.exists(OBJAVERSE_ANNOTATIONS_PATH):
            raise FileNotFoundError(
                f'Could not find objaverse annotations at {OBJAVERSE_ANNOTATIONS_PATH}.'
                f' Please follow the instructions in the README.md to download the annotations.'
            )
        _OBJAVERSE_ANNOTATIONS = compress_json.load(OBJAVERSE_ANNOTATIONS_PATH)
    return _OBJAVERSE_ANNOTATIONS
