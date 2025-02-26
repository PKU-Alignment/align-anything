import os
import glob
import shutil
import pathlib
import ai2thor.fifo_server
import ai2thor.platform
from objathor.asset_conversion.util import (
    view_asset_in_thor,
    get_existing_thor_asset_file_path,
    make_asset_pahts_relative,
    load_existing_thor_asset_file,
    save_thor_asset_file,
    EXTENSIONS_LOADABLE_IN_UNITY,
)
import ai2thor.controller
from ai2thor.hooks.procedural_asset_hook import ProceduralAssetHookRunner


def convert_to_unity_loadable(
    asset_source_dir,
    asset_target_dir,
    delete_original=False,
    replace_if_exists=False,
    asset_ids=[],
    target_extension=".msgpack.gz",
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
        if os.path.exists(asset_target_dir_path):
            print(f"Asset {id} already exists in target, skipping")
            continue

        try:
            existing_asset_file_in_source = get_existing_thor_asset_file_path(
                asset_source_dir_path, id
            )
        except RuntimeError:
            print(f"Asset {id} does not exist in source, skipping")
            continue

        try:
            existing_in_target_with_target_extension = get_existing_thor_asset_file_path(
                asset_target_dir_path, id, force_extension=target_extension
            )
            if not replace_if_exists:
                print(
                    f"Target {existing_in_target_with_target_extension} of extension `{target_extension}` already exists, skipping"
                )
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

        print(f"Normalizing texture path of {id}")
        pre_asset = load_existing_thor_asset_file(asset_source_dir_path, id)

        # to not do it again if already done
        # if os.path.isabs(pre_asset["normalTexturePath"]) or os.path.isabs(pre_asset["albedoTexturePath"]) or os.path.isabs(pre_asset["emissionTexturePath"]):
        print(f"Pre asset albedo: {pre_asset['albedoTexturePath']}")
        asset = make_asset_pahts_relative(pre_asset)
        print(f"Post asset albedo: {asset['albedoTexturePath']}")
        save_asset_path = os.path.join(asset_target_dir_path, f"{id}{target_extension}")
        save_thor_asset_file(asset, save_asset_path)
        print(f"Wrote asset to `{save_asset_path}`")

        # Not sure if we want to delete original if asset_source_dir == asset_target_dir? might lose data so did  not do it
        # if delete_original:
        #     print(f"Deleting original `{existing_asset_file_in_source}`")
        #     os.remove(existing_asset_file_in_source)


if __name__ == "__main__":
    # asset_source_dir = "/Users/alvaroh/net/nfs.cirrascale/prior/datasets/vida_datasets/objaverse_vida/processed_2023_07_28"
    # asset_target_dir = "/Users/alvaroh/net/nfs.cirrascale/prior/datasets/vida_datasets/objaverse_vida/test"
    # asset_ids = ['3203a050e9e647339340b64c8b38751f']
    asset_source_dir = "/net/nfs/prior/datasets/vida_datasets/objaverse_assets/2023_07_28/assets"
    asset_target_dir = "/net/nfs/prior/khzeng/objaverse_assets/2024_04_29/assets"
    asset_ids = os.listdir(asset_source_dir)
    convert_to_unity_loadable(
        asset_source_dir=asset_source_dir,
        asset_target_dir=asset_target_dir,
        asset_ids=asset_ids,
        delete_original=True,
        replace_if_exists=True,
    )

    ## OPTIONAL for testing the asset works and visualizing it
    hook_runner = ProceduralAssetHookRunner(
        asset_directory=asset_source_dir,
        asset_symlink=True,
        verbose=True,
        asset_limit=200,
        extension=".msgpack.gz",
    )

    controller = ai2thor.controller.Controller(
        # local_executable_path="unity/builds/thor-OSXIntel64-local/thor-OSXIntel64-local.app/Contents/MacOS/AI2-THOR",
        # local_build=True,
        agentMode="stretch",
        scene="Procedural",
        gridSize=0.25,
        width=300,
        height=300,
        visibilityScheme="Distance",
        action_hook_runner=hook_runner,
        commit_id="ca27b6ffa6881de3711e3b8cce184c8b1111a325",
        server_class=ai2thor.fifo_server.FifoServer,
        platform=ai2thor.platform.CloudRendering,
    )

    view_asset_in_thor(
        asset_id=asset_ids[0],
        controller=controller,
        output_dir=os.path.join(asset_target_dir, asset_ids[0], "thor_renders"),
        rotations=[
            (x, y, z, degrees) for degrees in [0, 90, 180, 270] for (x, y, z) in ((0, 1, 0),)
        ],
    )
