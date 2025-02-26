import os
import glob
import shutil
import pathlib
import argparse
import sys

import ai2thor.fifo_server
from objathor.asset_conversion.util import (
    view_asset_in_thor,
    get_existing_thor_asset_file_path,
    load_existing_thor_asset_file,
    save_thor_asset_file,
    EXTENSIONS_LOADABLE_IN_UNITY,
)
import ai2thor.controller
from ai2thor.hooks.procedural_asset_hook import ProceduralAssetHookRunner


def make_assets_paths_relative(asset, save_dir="."):
    asset["albedoTexturePath"] = os.path.join(
        save_dir,
        os.path.basename(asset["albedoTexturePath"]),
    )
    if "metallicSmoothnessTexturePath" in asset:
        asset["metallicSmoothnessTexturePath"] = os.path.join(
            save_dir,
            os.path.basename(asset["metallicSmoothnessTexturePath"]),
        )
    asset["normalTexturePath"] = os.path.join(
        save_dir,
        os.path.basename(asset["normalTexturePath"]),
    )
    if "emissionTexturePath" in asset:
        asset["emissionTexturePath"] = os.path.join(
            save_dir,
            os.path.basename(asset["emissionTexturePath"]),
        )
    return asset


def convert_to_unity_loadable(
    asset_source_dir,
    asset_target_dir,
    delete_original=False,
    replace_if_exists=False,
    asset_ids=[],
    target_extension=".msgpack.gz",
    source_extension=None,
):
    if target_extension not in EXTENSIONS_LOADABLE_IN_UNITY:
        raise Exception(
            f"Invalid extention `{target_extension}` must be one of {EXTENSIONS_LOADABLE_IN_UNITY}"
        )
    # no asset_ids in args means all assets
    if len(asset_ids) == 0:
        asset_paths = glob.glob(f"{os.path.join(asset_source_dir, '*')}")
    else:
        asset_paths = [os.path.join(asset_source_dir, id) for id in asset_ids]
    # print(asset_paths)
    for asset_source_dir_path in asset_paths:

        id = os.path.basename(os.path.normpath(asset_source_dir_path))
        asset_target_dir_path = os.path.join(asset_target_dir, id)

        try:
            existing_asset_file_in_source = get_existing_thor_asset_file_path(
                asset_source_dir_path, id, force_extension=source_extension
            )
        except:
            print(asset_source_dir_path)
            continue
        existing_in_target_with_target_extension = None

        try:
            existing_in_target_with_target_extension = get_existing_thor_asset_file_path(
                asset_target_dir_path, id, force_extension=target_extension
            )
            if not replace_if_exists:
                # print(
                #     f"Target {existing_in_target_with_target_extension} of extension `{target_extension}` already exists, skipping")
                continue
        except Exception:
            # continue processing as target does not exist
            pass

        # if source is different to target then copy asset directory to target
        if asset_source_dir != asset_target_dir:
            ignore = shutil.ignore_patterns()
            if delete_original:
                ignore = shutil.ignore_patterns(
                    f"*{''.join(pathlib.Path(existing_asset_file_in_source).suffixes)}"
                )
            shutil.copytree(
                asset_source_dir_path,
                asset_target_dir_path,
                ignore=ignore,
                dirs_exist_ok=replace_if_exists,
            )

        # To save work if script failed for some reason
        # if existing_in_target_with_target_extension:
        #     try:
        #         target_asset = load_existing_thor_asset_file(asset_target_dir_path, id, force_extension=target_extension)
        #         if target_asset["albedoTexturePath"].startswith("./albedo"):
        #             print(f"skip {id}")
        #             continue
        #     except Exception:
        #         pass

        # print(f"Normalizing texture path of {id}")
        try:
            pre_asset = load_existing_thor_asset_file(
                asset_source_dir_path, id, force_extension=source_extension
            )
        except:
            print(f"Error loading asset {asset_source_dir_path}")
            continue

        # to not do it again if already done
        # if os.path.isabs(pre_asset["normalTexturePath"]) or os.path.isabs(pre_asset["albedoTexturePath"]) or os.path.isabs(pre_asset["emissionTexturePath"]):
        # print(f"Pre asset albedo: {pre_asset['albedoTexturePath']}")
        asset = make_assets_paths_relative(pre_asset)
        # print(f"Post asset albedo: {asset['albedoTexturePath']}")
        save_asset_path = os.path.join(asset_target_dir_path, f"{id}{target_extension}")
        save_thor_asset_file(asset, save_asset_path)
        # print(f"Wrote asset to `{save_asset_path}`")

        # Not sure if we want to delete original if asset_source_dir == asset_target_dir? might lose data so did  not do it
        # if delete_original:
        #     print(f"Deleting original `{existing_asset_file_in_source}`")
        #     os.remove(existing_asset_file_in_source)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--delete_with_source_extension", action="store_true", help="Deletes original file"
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Runs conversion even if file with target extension exists in asset_target_dir",
    )

    parser.add_argument("--asset_source_dir", type=str)

    parser.add_argument(
        "--target_extension",
        type=str,
        default=".msgpack.gz",
    )

    parser.add_argument(
        "--asset_target_dir",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--source_extension",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--asset_ids", type=str, default="", help="Comma separated string of hex ids"
    )

    args = parser.parse_args(sys.argv[1:])
    # asset_source_dir = "/Users/alvaroh/net/nfs.cirrascale/prior/datasets/vida_datasets/objaverse_vida/processed_2023_07_28"
    asset_source_dir = "/Users/alvaroh/net/nfs.cirrascale/prior/datasets/vida_datasets/objaverse_vida/processed_2023_07_28/"
    asset_target_dir = asset_source_dir
    # asset_target_dir = "/Users/alvaroh/net/nfs.cirrascale/prior/datasets/vida_datasets/objaverse_vida/test"

    asset_ids = []
    if args.asset_ids != "":
        asset_ids.split(",")

    asset_ids = []  # ["2119148f2741425aa4eec00f970b3720"]# ['3203a050e9e647339340b64c8b38751f']
    target_extension = ".msgpack.gz"
    convert_to_unity_loadable(
        asset_source_dir=args.asset_source_dir,
        asset_target_dir=args.asset_target_dir if args.asset_target_dir else args.asset_source_dir,
        asset_ids=asset_ids,
        delete_original=args.delete_with_source_extension,
        replace_if_exists=args.force,
        target_extension=args.target_extension,
        source_extension=args.source_extension,
    )

    ## OPTIONAL for testing the asset works and visualizing it
    # hook_runner = ProceduralAssetHookRunner(
    #     asset_directory=asset_source_dir,
    #     asset_symlink=True,
    #     verbose=True,
    #     asset_limit=200,
    #     extension=target_extension,
    #     load_file_in_unity=True
    #     # extension=".msgpack.gz"

    # )

    # controller = ai2thor.controller.Controller(
    #     # local_executable_path="unity/builds/thor-OSXIntel64-local/thor-OSXIntel64-local.app/Contents/MacOS/AI2-THOR",
    #     # local_build=True,
    #     agentMode="stretch",
    #     scene="Procedural",
    #     gridSize=0.25,
    #     width=300,
    #     height=300,
    #     visibilityScheme="Distance",
    #     action_hook_runner=hook_runner,

    #     commit_id="150d4d18801e62121ff04313875373913b4df360",
    #     server_class=ai2thor.fifo_server.FifoServer
    # )

    # view_asset_in_thor(
    #     asset_id=asset_ids[0],
    #     controller=controller,
    #     output_dir=os.path.join(asset_target_dir, asset_ids[0], "thor_renders"),
    #     rotations=[(x, y, z, degrees) for degrees in [0, 90, 180, 270] for (x, y, z) in ((0, 1, 0),)],
    # )
