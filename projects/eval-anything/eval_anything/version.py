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

# Copyright 2024 PKU-Alignment Team. All Rights Reserved.
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
"""The eval anything project."""

from typing import Dict, Tuple


__version__ = '0.0.1.dev0'
__license__ = 'Apache License, Version 2.0'
__author__ = 'PKU-Alignment Team'
__release__ = False


def get_version_info() -> Dict[str, str]:
    """Get detailed version information.

    Returns:
        Dict containing version, license, author and status information.
    """
    return {
        'version': __version__,
        'license': __license__,
        'author': __author__,
        'status': 'release' if __release__ else 'development',
    }


def parse_version() -> Tuple[int, ...]:
    """Parse version string into tuple of integers.

    Returns:
        Tuple of version numbers (major, minor, patch).
    """
    # Remove development suffix if present
    version = __version__.split('.dev')[0]

    # Remove beta/alpha suffix if present
    for suffix in ['b', 'a', 'rc']:
        if suffix in version:
            version = version.split(suffix)[0]
            break

    # Split version string and convert to integers
    try:
        return tuple(map(int, version.split('.')))
    except ValueError:
        print(f'Failed to parse version number: {version}')
        return (0, 0, 0)


def get_version_string() -> str:
    """Get formatted version string with status.

    Returns:
        Formatted version string.
    """
    status = 'Release' if __release__ else 'Development'
    return f'{__version__} ({status})'


if not __release__:
    import os
    import subprocess

    try:
        prefix, sep, suffix = (
            subprocess.check_output(
                ['git', 'describe', '--abbrev=7'],  # noqa: S603,S607
                cwd=os.path.dirname(os.path.abspath(__file__)),
                stderr=subprocess.DEVNULL,
                text=True,
            )
            .strip()
            .lstrip('v')
            .replace('-', '.dev', 1)
            .replace('-', '+', 1)
            .partition('.dev')
        )
        if sep:
            version_prefix, dot, version_tail = prefix.rpartition('.')
            prefix = f'{version_prefix}{dot}{int(version_tail) + 1}'
            __version__ = sep.join((prefix, suffix))
            del version_prefix, dot, version_tail
        else:
            __version__ = prefix
        del prefix, sep, suffix
    except (OSError, subprocess.CalledProcessError):
        pass

    del os, subprocess


def check_version_compatibility(required_version: str) -> bool:
    """Check if current version meets the required version.

    Args:
        required_version: Minimum required version string.

    Returns:
        bool: True if current version meets requirement.
    """
    current = parse_version()
    required = tuple(map(int, required_version.split('.')))
    return current >= required[: len(current)]
