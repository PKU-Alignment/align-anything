# Copyright 2024 Allen Institute for AI
# ==============================================================================

import os
import warnings


ASSETS_VERSION = '2023_07_28'

OBJAVERSE_DATA_DIR = os.path.abspath(
    os.environ.get('OBJAVERSE_DATA_DIR', os.path.expanduser('~/.objathor-assets'))
)

if not os.path.basename(OBJAVERSE_DATA_DIR) == ASSETS_VERSION:
    OBJAVERSE_DATA_DIR = os.path.join(OBJAVERSE_DATA_DIR, ASSETS_VERSION)

OBJAVERSE_ASSETS_DIR = os.environ.get(
    'OBJAVERSE_ASSETS_DIR', os.path.join(OBJAVERSE_DATA_DIR, 'assets')
)
OBJAVERSE_ANNOTATIONS_PATH = os.environ.get(
    'OBJAVERSE_ANNOTATIONS_PATH', os.path.join(OBJAVERSE_DATA_DIR, 'annotations.json.gz')
)

for var_name in ['OBJAVERSE_ASSETS_DIR', 'OBJAVERSE_ANNOTATIONS_PATH']:
    if locals()[var_name] is None:
        warnings.warn(f'{var_name} is not set.')
    else:
        locals()[var_name] = os.path.abspath(locals()[var_name])


print(
    f'Using'
    f" '{OBJAVERSE_ASSETS_DIR}' for objaverse assets,"
    f" '{OBJAVERSE_ANNOTATIONS_PATH}' for objaverse annotations,"
)
