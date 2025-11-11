# Copyright 2024 Allen Institute for AI
# ==============================================================================

import os
from typing import Any, Dict, Optional

import compress_json

from eval_anything.third_party.SPOC.utils.constants.objaverse_data_dirs import (
    OBJAVERSE_ANNOTATIONS_PATH,
)


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
