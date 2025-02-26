from typing import Any, Dict, List, Sequence, Optional, Set

import gym
import numpy as np
from PIL import Image
from allenact.base_abstractions.sensor import Sensor

from utils.local_logging import unnormalize_image, WandbLoggingSensor

import wandb
from allenact.base_abstractions.callbacks import Callback
import os


class SimpleWandbLogging(Callback):
    def __init__(
        self,
        project: str,
        entity: str,
    ):
        self.project = project
        self.entity = entity

        self._defined_metrics: Set[str] = set()

    def setup(self, name: str, **kwargs) -> None:
        if "OUTPUT_DIR" in os.environ.keys() and "EXTRA_TAG" in os.environ.keys():
            possible_wandb_id = "{}/used_configs/{}.txt".format(
                os.environ["OUTPUT_DIR"], os.environ["EXTRA_TAG"]
            )
            if os.path.isfile(possible_wandb_id):
                with open(possible_wandb_id, "r") as f:
                    wandb_id = f.read()
                wandb_id = wandb_id.split("\n")[0]
            else:
                wandb_id = wandb.util.generate_id()
                with open(possible_wandb_id, "w") as f:
                    f.write(wandb_id)
            wandb.init(
                project=self.project,
                entity=self.entity,
                name=name,
                config=kwargs,
                id=wandb_id,
                resume="allow",
            )
        else:
            wandb.init(
                project=self.project,
                entity=self.entity,
                name=name,
                config=kwargs,
            )

    def _define_missing_metrics(
        self,
        metric_means: Dict[str, float],
        scalar_name_to_total_experiences_key: Dict[str, str],
    ):
        for k, v in metric_means.items():
            if k not in self._defined_metrics:
                wandb.define_metric(
                    k,
                    step_metric=scalar_name_to_total_experiences_key.get(k, "training_step"),
                )

                self._defined_metrics.add(k)

    def on_train_log(
        self,
        metrics: List[Dict[str, Any]],
        metric_means: Dict[str, float],
        step: int,
        tasks_data: List[Any],
        scalar_name_to_total_experiences_key: Dict[str, str],
        **kwargs,
    ) -> None:
        """Log the train metrics to wandb."""

        self._define_missing_metrics(
            metric_means=metric_means,
            scalar_name_to_total_experiences_key=scalar_name_to_total_experiences_key,
        )

        wandb.log(
            {
                **metric_means,
                "training_step": step,
            }
        )

    def combine_rgb_across_episode(self, observation_list):
        all_rgb = []
        for obs in observation_list:
            frame = unnormalize_image(obs["rgb"])
            all_rgb.append(np.array(Image.fromarray((frame * 255).astype(np.uint8))))

        return all_rgb

    def on_valid_log(
        self,
        metrics: Dict[str, Any],
        metric_means: Dict[str, float],
        checkpoint_file_name: str,
        tasks_data: List[Any],
        scalar_name_to_total_experiences_key: Dict[str, str],
        step: int,
        **kwargs,
    ) -> None:
        """Log the validation metrics to wandb."""

        self._define_missing_metrics(
            metric_means=metric_means,
            scalar_name_to_total_experiences_key=scalar_name_to_total_experiences_key,
        )

        wandb.log(
            {
                **metric_means,
                "training_step": step,
            }
        )

    def get_table_content(self, metrics, tasks_data, frames_with_logit_flag=False):
        observation_list = [
            tasks_data[i]["local_logging_callback_sensor"]["observations"]
            for i in range(len(tasks_data))
        ]  # NOTE: List of episode frames

        list_of_video_frames = [self.combine_rgb_across_episode(obs) for obs in observation_list]

        path_list = [
            tasks_data[i]["local_logging_callback_sensor"]["path"] for i in range(len(tasks_data))
        ]  # NOTE: List of path frames

        if frames_with_logit_flag:
            frames_with_logits_list_numpy = [
                tasks_data[i]["local_logging_callback_sensor"]["frames_with_logits"]
                for i in range(len(tasks_data))
            ]

        table_content = []
        frames_with_logits_list = []

        videos_without_logits_list = []

        for idx, data in enumerate(zip(list_of_video_frames, path_list, metrics["tasks"])):
            frames_without_logits, path, metric_data = (
                data[0],
                data[1],
                data[2],
            )

            wandb_data = (
                wandb.Video(
                    np.moveaxis(np.array(frames_without_logits), [0, 3, 1, 2], [0, 1, 2, 3]),
                    fps=10,
                    format="mp4",
                ),
                wandb.Image(path[0]),
                metric_data["ep_length"],
                metric_data["success"],
                metric_data["dist_to_target"],
                metric_data["task_info"]["task_type"],
                metric_data["task_info"]["house_name"],
                metric_data["task_info"]["target_object_type"],
                metric_data["task_info"]["id"],
                idx,
            )

            videos_without_logits_list.append(
                wandb.Video(
                    np.moveaxis(np.array(frames_without_logits), [0, 3, 1, 2], [0, 1, 2, 3]),
                    fps=5,
                    format="mp4",
                ),
            )

            table_content.append(wandb_data)
            if frames_with_logit_flag:
                frames_with_logits_list.append(
                    wandb.Video(np.array(frames_with_logits_list_numpy[idx]), fps=10, format="mp4")
                )

        return table_content, frames_with_logits_list, videos_without_logits_list

    def on_test_log(
        self,
        checkpoint_file_name: str,
        metrics: Dict[str, Any],
        metric_means: Dict[str, float],
        tasks_data: List[Any],
        scalar_name_to_total_experiences_key: Dict[str, str],
        step: int,
        **kwargs,
    ) -> None:
        """Log the test metrics to wandb."""

        self._define_missing_metrics(
            metric_means=metric_means,
            scalar_name_to_total_experiences_key=scalar_name_to_total_experiences_key,
        )

        if tasks_data[0]["local_logging_callback_sensor"] is not None:
            frames_with_logits_flag = False

            if "frames_with_logits" in tasks_data[0]["local_logging_callback_sensor"]:
                frames_with_logits_flag = True

            (
                table_content,
                frames_with_logits_list,
                videos_without_logit_list,
            ) = self.get_table_content(metrics, tasks_data, frames_with_logits_flag)

            video_dict = {"all_videos": {}}

            for vid, data in zip(frames_with_logits_list, table_content):
                idx = data[-1]
                video_dict["all_videos"][idx] = vid

            table = wandb.Table(
                columns=[
                    "Trajectory",
                    "Path",
                    "Episode Length",
                    "Success",
                    "Dist to target",
                    "Task Type",
                    "House Name",
                    "Target Object Type",
                    "Task Id",
                    "Index",
                ]
            )

            for data in table_content:
                table.add_data(*data)

            wandb.log(
                {
                    **metric_means,
                    "training_step": step,
                    "Qualitative Examples": table,
                    "Videos": video_dict,
                }
            )
        else:
            wandb.log(
                {
                    **metric_means,
                    "training_step": step,
                    # "Qualitative Examples": table,
                    # "Videos": video_dict,
                }
            )

    def after_save_project_state(self, base_dir: str) -> None:
        pass

    def callback_sensors(self) -> Optional[Sequence[Sensor]]:
        return [
            WandbLoggingSensor(
                uuid="local_logging_callback_sensor", observation_space=gym.spaces.Discrete(1)
            ),
        ]
