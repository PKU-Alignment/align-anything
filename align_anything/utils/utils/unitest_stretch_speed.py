import os
import random
import time
import prior
import ai2thor
import ai2thor.platform
import torch.cuda
import gzip
import argparse
import json
from ai2thor.controller import Controller
from ai2thor.hooks.procedural_asset_hook import (
    ProceduralAssetHookRunner,
    get_all_asset_ids_recursively,
    create_assets_if_not_exist,
)
from environment.stretch_controller import StretchController
from utils.constants.stretch_initialization_utils import HORIZON
from utils.type_utils import THORActions, AgentPose, Vector3
from multiprocessing import Process
from tqdm import tqdm

try:
    from prior import LazyJsonDataset
except:
    raise ImportError("Please update the prior package (pip install --upgrade prior).")


ACTION = [
    "MoveAhead",
    "RotateRight",
    "RotateLeft",
]

ACTIONStretch = [
    THORActions.move_ahead,
    THORActions.move_back,
    THORActions.rotate_right,
    THORActions.rotate_left,
    THORActions.rotate_right_small,
    THORActions.rotate_left_small,
]


class ProceduralAssetHookRunnerResetOnNewHouse(ProceduralAssetHookRunner):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.last_asset_id_set = set()

    def Initialize(self, action, controller):
        if self.asset_limit > 0:
            return controller.step(
                action="DeleteLRUFromProceduralCache", assetLimit=self.asset_limit
            )

    def CreateHouse(self, action, controller):
        house = action["house"]
        asset_ids = get_all_asset_ids_recursively(house["objects"], [])
        asset_ids_set = set(asset_ids)
        if not asset_ids_set.issubset(self.last_asset_id_set):
            controller.step(action="DeleteLRUFromProceduralCache", assetLimit=0)
            self.last_asset_id_set = set(asset_ids)

        return create_assets_if_not_exist(
            controller=controller,
            asset_ids=asset_ids,
            asset_directory=self.asset_directory,
            asset_symlink=self.asset_symlink,
            stop_if_fail=self.stop_if_fail,
            copy_to_dir=os.path.join(controller._build.base_dir, self.target_dir),
            load_file_in_unity=True,
        )


def export_objaverse_assets():
    import os
    import warnings

    ASSETS_VERSION = "2023_07_28"
    if torch.cuda.is_available():
        OBJAVERSE_DATA_DIR = "/net/nfs/prior/datasets/vida_datasets/objaverse_assets/2023_07_28"
    else:
        OBJAVERSE_DATA_DIR = "/Users/khzeng/Desktop/objaverse_vida/2023_07_28"

    if not os.path.basename(OBJAVERSE_DATA_DIR) == ASSETS_VERSION:
        OBJAVERSE_DATA_DIR = os.path.join(OBJAVERSE_DATA_DIR, ASSETS_VERSION)

    OBJAVERSE_ASSETS_DIR = os.environ.get(
        "OBJAVERSE_ASSETS_DIR", os.path.join(OBJAVERSE_DATA_DIR, "assets")
    )
    if torch.cuda.is_available():
        OBJAVERSE_HOUSES_DIR = (
            "/net/nfs/prior/datasets/vida_datasets/objaverse_vida/houses_2023_07_28"
        )
    else:
        OBJAVERSE_HOUSES_DIR = "/Users/khzeng/Desktop/objaverse_vida/houses_2023_07_28"

    for var_name in ["OBJAVERSE_ASSETS_DIR", "OBJAVERSE_HOUSES_DIR"]:
        if locals()[var_name] is None:
            warnings.warn(f"{var_name} is not set.")
        else:
            locals()[var_name] = os.path.abspath(locals()[var_name])

    if OBJAVERSE_HOUSES_DIR is None:
        warnings.warn("`OBJAVERSE_HOUSES_DIR` is not set.")
    else:
        OBJAVERSE_HOUSES_DIR = os.path.abspath(OBJAVERSE_HOUSES_DIR)

    print(
        f"Using"
        f" '{OBJAVERSE_ASSETS_DIR}' for objaverse assets,"
        f" '{OBJAVERSE_HOUSES_DIR}' for procthor-objaverse houses."
    )
    return OBJAVERSE_ASSETS_DIR, OBJAVERSE_HOUSES_DIR


def split_to_tasks(file_base_name):
    data = {}
    for split in ("train", "val"):
        if split == "train":
            filename = "{}/train.jsonl.gz".format(file_base_name)
        else:
            filename = "{}/val.jsonl.gz".format(file_base_name)
        if not os.path.exists(filename):
            print(f"Error: ObjectNavLocalRef for {split} not found")
            return []

        with gzip.open(filename, "rt") as f:
            tasks = [line for line in tqdm(f, desc=f"Loading {split}")]
        data[split] = LazyJsonDataset(data=tasks, dataset="ObjectNavLocalRef", split=split)
    tasks = prior.DatasetDict(**data)

    # This is to make the evaluation consistent
    tasks.val = prior.load_dataset(
        dataset="spoc-data",
        entity="spoc-robot",
        revision="chores-small",
        task_types=["ObjectNavLocalRef"],
    ).val
    return tasks


def speed_test_procthor(pid=0, mod=False, total_p=1):
    on_server = torch.cuda.is_available()

    if on_server:
        house = prior.load_dataset(
            dataset="procthor-100k", entity="roseh-ai2", revision="100k-balanced-may03"
        )
    else:
        house = prior.load_dataset(dataset="procthor-100k", entity="roseh-ai2", revision="tiny")

    STRETCH_BUILD_ID = "966bd7758586e05d18f6181f459c0e90ba318bec"
    if on_server:
        device = pid % 8 if mod else 0
    else:
        device = None
    STRETCH_ENV_ARGS = dict(
        gridSize=0.2,
        width=224,
        height=224,
        visibilityDistance=1.0,
        fieldOfView=59,
        server_class=ai2thor.fifo_server.FifoServer,
        useMassThreshold=True,
        massThreshold=10,
        autoSimulation=False,
        autoSyncTransforms=True,
        agentMode="stretch",
        renderInstanceSegmentation=True,
        renderDepthImage=True,
        cameraNearPlane=0.01,
        commit_id=STRETCH_BUILD_ID,
        server_timeout=10000,
        snapToGrid=False,
        platform=ai2thor.platform.CloudRendering if on_server else None,
        gpu_device=device,
        render_mani_camera=False,
        use_quick_navi_action=True,
    )
    c = Controller(**STRETCH_ENV_ARGS)
    scene = random.choice(house["train"])
    c.reset(scene=scene)
    t_start = time.time()
    for i in range(512):
        a = "MoveAhead"
        c.step(a)
    t_end = time.time()
    print(f"512 steps, estimated time in python {t_end - t_start}")


def speed_test_objathor(pid=0, mod=False, total_p=1):
    on_server = torch.cuda.is_available()

    OBJAVERSE_ASSETS_DIR, OBJAVERSE_HOUSES_DIR = export_objaverse_assets()
    _ACTION_HOOK_RUNNER = ProceduralAssetHookRunnerResetOnNewHouse(
        asset_directory=OBJAVERSE_ASSETS_DIR, asset_symlink=True, verbose=True, asset_limit=200
    )

    if on_server:
        max_houses_per_split = {"train": 1000, "val": 0, "test": 0}
    else:
        max_houses_per_split = {"train": 64, "val": 0, "test": 0}

    house = prior.load_dataset(
        dataset="spoc-data",
        entity="spoc-robot",
        revision="local-objaverse-procthor-houses",
        path_to_splits=None,
        split_to_path={
            k: os.path.join(OBJAVERSE_HOUSES_DIR, f"{k}.jsonl.gz") for k in ["train", "val", "test"]
        },
        max_houses_per_split=max_houses_per_split,
    )

    STRETCH_BUILD_ID = "966bd7758586e05d18f6181f459c0e90ba318bec"
    if on_server:
        device = pid % 8 if mod else 0
    else:
        device = None
    STRETCH_ENV_ARGS = dict(
        gridSize=0.2 * 0.75,
        width=396,
        height=224,
        visibilityDistance=1.0,
        fieldOfView=59,
        server_class=ai2thor.fifo_server.FifoServer,
        useMassThreshold=True,
        massThreshold=10,
        autoSimulation=False,
        autoSyncTransforms=True,
        agentMode="stretch",
        renderInstanceSegmentation=True,
        renderDepthImage=True,
        cameraNearPlane=0.01,
        commit_id=STRETCH_BUILD_ID,
        server_timeout=10000,
        snapToGrid=False,
        fastActionEmit=True,
        action_hook_runner=_ACTION_HOOK_RUNNER,
        platform=ai2thor.platform.CloudRendering if on_server else None,
        gpu_device=device,
        render_mani_camera=False,
        use_quick_navi_action=True,
    )
    c = StretchController(**STRETCH_ENV_ARGS)
    scene = random.choice(house["train"])
    c.reset(scene=scene)
    c.controller.step("Pass")
    t_start = time.time()
    for i in range(100):
        a = random.choice(ACTIONStretch)
        c.agent_step(a)
    t_end = time.time()
    print(f"100 steps, estimated time in python {t_end - t_start}")


def speed_test_locobot_objathor(pid=0, mod=False, total_p=1):
    on_server = torch.cuda.is_available()

    OBJAVERSE_ASSETS_DIR, OBJAVERSE_HOUSES_DIR = export_objaverse_assets()
    _ACTION_HOOK_RUNNER = ProceduralAssetHookRunnerResetOnNewHouse(
        asset_directory=OBJAVERSE_ASSETS_DIR, asset_symlink=True, verbose=True, asset_limit=200
    )

    if on_server:
        max_houses_per_split = {"train": 1000, "val": 0, "test": 0}
    else:
        max_houses_per_split = {"train": 64, "val": 0, "test": 0}

    house = prior.load_dataset(
        dataset="spoc-data",
        entity="spoc-robot",
        revision="local-objaverse-procthor-houses",
        path_to_splits=None,
        split_to_path={
            k: os.path.join(OBJAVERSE_HOUSES_DIR, f"{k}.jsonl.gz") for k in ["train", "val", "test"]
        },
        max_houses_per_split=max_houses_per_split,
    )

    STRETCH_BUILD_ID = "966bd7758586e05d18f6181f459c0e90ba318bec"
    if on_server:
        device = pid % 8 if mod else 0
    else:
        device = None
    ENV_ARGS = dict(
        gridSize=0.2 * 0.75,
        width=396,
        height=224,
        visibilityDistance=1.0,
        fieldOfView=59,
        server_class=ai2thor.fifo_server.FifoServer,
        useMassThreshold=True,
        massThreshold=10,
        autoSimulation=False,
        autoSyncTransforms=True,
        agentMode="locobot",
        renderInstanceSegmentation=True,
        renderDepthImage=True,
        cameraNearPlane=0.01,
        commit_id=STRETCH_BUILD_ID,
        server_timeout=10000,
        snapToGrid=False,
        fastActionEmit=True,
        action_hook_runner=_ACTION_HOOK_RUNNER,
        platform=ai2thor.platform.CloudRendering if on_server else None,
        gpu_device=device,
    )
    c = Controller(**ENV_ARGS)
    scene = random.choice(house["train"])
    c.reset(scene=scene)
    c.step("Pass")
    pose = scene["metadata"]["agent"].copy()
    del pose["standing"]
    c.step(action="TeleportFull", **pose)
    c.step(action="Pass")
    c.step(action="GetReachablePositions")
    assert len(c.last_event.metadata["actionReturn"]) > 0
    t_start = time.time()
    for i in range(100):
        a = random.choice(ACTION)
        c.step(a)
    t_end = time.time()
    print(f"100 steps, estimated time in python {t_end - t_start}")


def speed_test_objathor_on_ObjectNavLocalRef(pid=0, mod=False, total_p=1):
    on_server = torch.cuda.is_available()

    OBJAVERSE_ASSETS_DIR, OBJAVERSE_HOUSES_DIR = export_objaverse_assets()
    _ACTION_HOOK_RUNNER = ProceduralAssetHookRunnerResetOnNewHouse(
        asset_directory=OBJAVERSE_ASSETS_DIR, asset_symlink=True, verbose=True, asset_limit=200
    )

    if on_server:
        max_houses_per_split = {"train": 150000, "val": 0, "test": 0}
        task_base_dir = "/net/nfs/prior/datasets/vida_datasets/jhu_rl/ObjectNavLocalRef/"
    else:
        max_houses_per_split = {"train": 5000, "val": 0, "test": 0}
        task_base_dir = "/Users/khzeng/Desktop/ckpts/ObjectNavLocalRef"

    house = prior.load_dataset(
        dataset="spoc-data",
        entity="spoc-robot",
        revision="local-objaverse-procthor-houses",
        path_to_splits=None,
        split_to_path={
            k: os.path.join(OBJAVERSE_HOUSES_DIR, f"{k}.jsonl.gz") for k in ["train", "val", "test"]
        },
        max_houses_per_split=max_houses_per_split,
    )
    tasks = split_to_tasks(task_base_dir).train
    per_house = max_houses_per_split["train"] // total_p
    # all_house = list(range(pid * per_house, (pid + 1) * per_house))
    all_house = [4711]
    tasks = [t for t in tasks if t["house_index"] in all_house]

    STRETCH_BUILD_ID = "966bd7758586e05d18f6181f459c0e90ba318bec"
    if on_server:
        device = pid % 8 if mod else 0
    else:
        device = None
    STRETCH_ENV_ARGS = dict(
        gridSize=0.2 * 0.75,
        width=396,
        height=224,
        visibilityDistance=1.0,
        fieldOfView=59,
        server_class=ai2thor.fifo_server.FifoServer,
        useMassThreshold=True,
        massThreshold=10,
        autoSimulation=False,
        autoSyncTransforms=True,
        agentMode="stretch",
        renderInstanceSegmentation=True,
        renderDepthImage=True,
        cameraNearPlane=0.01,
        commit_id=STRETCH_BUILD_ID,
        server_timeout=10000,
        snapToGrid=False,
        fastActionEmit=True,
        action_hook_runner=_ACTION_HOOK_RUNNER,
        platform=ai2thor.platform.CloudRendering if on_server else None,
        gpu_device=device,
        render_mani_camera=False,
        use_quick_navi_action=True,
    )
    c = StretchController(**STRETCH_ENV_ARGS)
    scene = random.choice(house["train"])
    c.reset(scene=scene)
    c.controller.step("Pass")
    c.teleport_agent(**scene["metadata"]["agent"])
    c.controller.step("Pass")
    last_t = tasks[0]
    failed_house = []
    for idx, t in enumerate(tasks):
        if idx == 0 or last_t["house_index"] != t["house_index"]:
            scene = house["train"][t["house_index"]]
            c.reset(scene=scene)
            # c.teleport_agent(**scene["metadata"]["agent"])
            # c.controller.step("Pass")
        last_t = t
        agent_pose = AgentPose(
            position=Vector3(
                x=t["agent_starting_position"][0],
                y=t["agent_starting_position"][1],
                z=t["agent_starting_position"][2],
            ),
            rotation=Vector3(x=0, y=t["agent_y_rotation"], z=0),
            horizon=HORIZON,
            standing=True,
        )
        c.teleport_agent(**agent_pose)
        if not c.controller.last_event.metadata["lastActionSuccess"]:
            print("house {} failed".format(t["house_index"]))
            failed_house.append(t["house_index"])
        else:
            c.controller.step("Pass")
            print("house {} passed".format(t["house_index"]))
    json.dump(failed_house, open("failed_house_{}.json".format(pid), "w"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="profiler",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-n",
        "--num_p",
        required=False,
        default=1,
        type=int,
        help="number of processes",
    )
    parser.add_argument(
        "-p",
        "--procthor",
        dest="procthor",
        action="store_true",
        required=False,
        help="use procthor or not",
    )
    parser.add_argument(
        "-l",
        "--locobot",
        dest="locobot",
        action="store_true",
        required=False,
        help="use LoCoBot or not",
    )
    parser.add_argument(
        "-m",
        "--mod",
        dest="mod",
        action="store_true",
        required=False,
        help="assign process to different GPUs uniformly",
    )
    args = parser.parse_args()

    if args.locobot:
        profile_func = speed_test_locobot_objathor
    else:
        # profile_func = speed_test_procthor if args.procthor else speed_test_objathor
        profile_func = (
            speed_test_procthor if args.procthor else speed_test_objathor_on_ObjectNavLocalRef
        )
    if args.num_p == 1:
        profile_func(0, args.mod)
    else:
        ps = []
        for pid in range(args.num_p):
            p = Process(target=profile_func, args=(pid, args.mod, args.num_p))
            ps.append(p)
            p.start()
        for p in ps:
            p.join()
