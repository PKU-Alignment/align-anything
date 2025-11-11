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

# Copyright 2023-2024 PKU-Alignment Team. All Rights Reserved.
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

import pathlib
import re
import sys

from setuptools import setup


HERE = pathlib.Path(__file__).absolute().parent
VERSION_FILE = HERE / 'eval_anything' / 'version.py'
sys.path.insert(0, str(VERSION_FILE.parent))

import version  # noqa


VERSION_CONTENT = None

try:
    if not version.__release__:
        try:
            VERSION_CONTENT = VERSION_FILE.read_text(encoding='utf-8')
            VERSION_FILE.write_text(
                data=re.sub(
                    r"""__version__\s*=\s*('[^']+'|"[^"]+")""",
                    f"__version__ = '{version.__version__}'",
                    string=VERSION_CONTENT,
                ),
                encoding='utf-8',
            )
        except OSError:
            VERSION_CONTENT = None

    setup(
        name='eval_anything',
        version=version.__version__,
        entry_points={
            'console_scripts': [
                'eval-anything-cli = eval_anything.cli:main',
            ],
        },
    )
finally:
    if VERSION_CONTENT is not None:
        with VERSION_FILE.open(mode='wt', encoding='utf-8', newline='') as file:
            file.write(VERSION_CONTENT)
