import argparse
import copy
import os
import subprocess
from multiprocessing import Pool
from tempfile import TemporaryDirectory

from objathor.utils.download_utils import download_with_progress_bar


def map_task_type(task_type: str) -> str:
    task_type_map = dict(SimpleExploreHouse='RoomVisit', ObjectNavOpenVocab='ObjectNavDescription')
    return task_type_map.get(task_type) or task_type


def inverse_map_task_type(task_type: str) -> str:
    task_type_map = dict(RoomVisit='SimpleExploreHouse', ObjectNavDescription='ObjectNavOpenVocab')
    return task_type_map.get(task_type) or task_type


def map_task_spec(task_spec: 'TaskSpec') -> 'TaskSpec':
    task_spec = copy.copy(task_spec)
    task_spec['task_type'] = map_task_type(task_spec['task_type'])
    return task_spec


ALL_TASK_TYPES = [
    'PickupType',
    'RoomVisit',
    'FetchType',
    'ObjectNavType',
    'RoomNav',
    'ObjectNavAffordance',
    'ObjectNavLocalRef',
    'ObjectNavDescription',
    'ObjectNavRelAttribute',
    'ObjectNavRoom',
]


def untar_file(tar_file: str, destination_dir: str):
    command = f'tar -xzf {tar_file} -C {destination_dir} --strip-components=1'
    print('Running:', command)
    subprocess.call(command, shell=True)


def download_and_untar_file(info):
    url = info['url']
    save_dir = info['save_dir']

    tar_parts = os.path.basename(url).split('.')
    dest_task_type = map_task_type(tar_parts[0])
    tar_name = '.'.join([dest_task_type] + tar_parts[1:])

    out_dir = os.path.join(save_dir, dest_task_type)
    os.makedirs(out_dir, exist_ok=True)

    local_dir_obj = TemporaryDirectory()
    with local_dir_obj as local_dir:
        tmp_save_path = os.path.join(local_dir, tar_name)
        download_with_progress_bar(
            url=url,
            save_path=tmp_save_path,
            desc=f'Downloading: {tar_name}.',
        )
        untar_file(tmp_save_path, out_dir)


def main():
    parser = argparse.ArgumentParser(description='Train dataset downloader.')
    parser.add_argument('--save_dir', required=True, help='Directory to save the downloaded files.')
    parser.add_argument('--types', required=True, help="Should be either 'fifteen' or 'all'.")
    parser.add_argument(
        '--task_types',
        default=None,
        nargs='+',
        help=f'List of task names to download, by default this will include all tasks. Should be a subset of: {ALL_TASK_TYPES}',
    )
    parser.add_argument('--num', '-n', default=1, type=int, help='Number of parallel downloads.')
    args = parser.parse_args()

    assert args.types in ['fifteen', 'all'], "Types should be either 'fifteen' or 'all'."

    if args.task_types is None:
        args.task_types = ALL_TASK_TYPES

    args.save_dir = os.path.abspath(os.path.expanduser(os.path.join(args.save_dir, args.types)))
    os.makedirs(args.save_dir, exist_ok=True)

    download_args = []
    for tn in args.task_types:
        orig_task_type = inverse_map_task_type(tn)
        download_args.append(
            dict(
                url=f'https://pub-bebbada739114fa1aa96aaf25c873a66.r2.dev/{args.types}_type/{orig_task_type}.tar.gz',
                save_dir=args.save_dir,
            )
        )

    with Pool(args.num) as pool:
        pool.map(download_and_untar_file, download_args)


if __name__ == '__main__':
    main()
