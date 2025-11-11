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

import argparse
import os
import subprocess
from tempfile import TemporaryDirectory

from objathor.utils.download_utils import download_with_progress_bar


def download_and_rename_file(info):
    url = info['url']
    save_dir = info['save_dir']
    subset = info['subset']

    local_dir_obj = TemporaryDirectory()
    with local_dir_obj as local_dir:
        tmp_save_path = os.path.join(local_dir, os.path.basename(url))
        download_with_progress_bar(
            url=url,
            save_path=tmp_save_path,
            desc=f"Downloading: {url.split('/')[-1]}.",
        )
        destination_folder = os.path.join(save_dir, 'houses_2023_07_28')
        os.makedirs(destination_folder, exist_ok=True)
        destination_dir = os.path.join(destination_folder, f'{subset}.jsonl.gz')
        command = f'mv {tmp_save_path} {destination_dir}'
        print('Running:', command)
        subprocess.call(command, shell=True)


def main():
    parser = argparse.ArgumentParser(description='Train dataset downloader.')
    parser.add_argument('--save_dir', required=True, help='Directory to save the downloaded files.')
    parser.add_argument('--subset', required=True, help="Should be either 'train' or 'val'.")

    args = parser.parse_args()

    assert args.subset in ['train', 'val'], "Should be either 'train' or 'val'."

    args.save_dir = os.path.abspath(os.path.expanduser(args.save_dir))
    os.makedirs(args.save_dir, exist_ok=True)

    if args.subset == 'train':
        data_link = 'https://pub-5932b61898254419952f5b13d42d82ab.r2.dev/procthor_objaverse%2F2023_07_28%2Ftrain.jsonl.gz'
    elif args.subset == 'val':
        data_link = 'https://pub-5932b61898254419952f5b13d42d82ab.r2.dev/procthor_objaverse%2F2023_07_28%2Fval.jsonl.gz'
    else:
        raise ValueError(f'Unknown subset: {args.subset}')

    download_and_rename_file(dict(url=data_link, save_dir=args.save_dir, subset=args.subset))


if __name__ == '__main__':
    main()
