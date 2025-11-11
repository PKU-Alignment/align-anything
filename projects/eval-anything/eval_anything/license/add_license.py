#!/usr/bin/env python3
import sys
from pathlib import Path


LICENSE_HEADER = '''\
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
'''


def should_skip(file: Path) -> bool:
    # 跳过空文件
    if file.stat().st_size == 0:
        print(f"Skipping empty file: {file}")
        return True

    return False


def add_license_if_missing(path: Path):
    if should_skip(path):
        return
    SKIP_PATHS = {
        'eval_anything/benchmarks/text_to_text/DoAnythingNow/utils.py',
        # 可继续添加其他路径
    }

    if str(path).replace('\\', '/') in SKIP_PATHS:
        print(f"Skipping blacklisted file: {path}")
        return
    try:
        content = path.read_text(encoding='utf-8')
    except Exception as e:
        print(f"Failed to read {path}: {e}")
        return

    if LICENSE_HEADER.strip() in content:
        return  # Already has license

    # Insert after shebang or encoding if present
    lines = content.splitlines()
    insert_at = 0
    if lines and lines[0].startswith('#!'):
        insert_at = 1
    elif lines and 'coding' in lines[0]:
        insert_at = 1

    new_content = lines[:insert_at] + LICENSE_HEADER.splitlines() + [''] + lines[insert_at:]
    try:
        path.write_text('\n'.join(new_content) + '\n', encoding='utf-8')
        print(f"Inserted license header in {path}")
    except Exception as e:
        print(f"Failed to write {path}: {e}")


def main():
    for file in sys.argv[1:]:
        p = Path(file)
        if p.suffix == '.py' or p.suffix == '.yaml':
            add_license_if_missing(p)


if __name__ == '__main__':
    main()
