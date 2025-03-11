# Copyright 2024 Allen Institute for AI

# Copyright 2024-2025 Align-Anything Team. All Rights Reserved.
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


import copy
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from align_anything.environment.stretch_controller import StretchController
from align_anything.utils.utils.constants.stretch_initialization_utils import stretch_long_names


DISTINCT_COLORS = [
    (255, 0, 0),  # Red
    (0, 255, 0),  # Green
    (0, 0, 255),  # Blue
    (255, 255, 0),  # Yellow
    (0, 255, 255),  # Cyan
    (255, 0, 255),  # Magenta
    (128, 0, 0),  # Dark Red
    (0, 128, 0),  # Dark Green
    (0, 0, 128),  # Dark Blue
    (128, 128, 0),  # Olive
    (128, 0, 128),  # Purple
    (0, 128, 128),  # Teal
    (192, 192, 192),  # Silver
    (128, 128, 128),  # Gray
    (255, 165, 0),  # Orange
    (255, 192, 203),  # Pink
    (255, 255, 255),  # White
    (0, 0, 0),  # Black
    (0, 0, 139),  # DarkBlue
    (0, 100, 0),  # DarkGreen
    (139, 0, 139),  # DarkMagenta
    (165, 42, 42),  # Brown
    (255, 215, 0),  # Gold
    (64, 224, 208),  # Turquoise
    (240, 230, 140),  # Khaki
    (70, 130, 180),  # Steel Blue
]


def add_bboxes_to_frame(
    frame: np.ndarray,
    bboxes: Sequence[Sequence[float]],
    labels: Optional[Sequence[str]],
    inplace=False,
    colors=tuple(DISTINCT_COLORS),
    thinkness=1,
):
    """
    Visualize bounding boxes on an image and save the image to disk.

    Parameters:
    - frame: numpy array of shape (height, width, 3) representing the image.
    - bboxes: list of bounding boxes. Each bounding box is a list of [min_row, min_col, max_row, max_col].
    - labels: list of labels corresponding to each bounding box.
    - inplace: whether to modify the input frame in place or return a new frame.
    """
    # Convert numpy image to PIL Image for visualization

    assert frame.dtype == np.uint8
    if not inplace:
        frame = copy.deepcopy(frame)

    bboxes_cleaned = [[int(v) for v in bbox] for bbox in bboxes if -1 not in bbox]
    if labels is None:
        labels = [''] * len(bboxes_cleaned)

    h, w, _ = frame.shape

    # Plot bounding boxes and labels
    for bbox, label, color in zip(bboxes_cleaned, labels, colors):
        if np.all(bbox == 0):
            continue
        cv2.rectangle(frame, bbox[:2], bbox[2:], color=color, thickness=thinkness)

        cv2.putText(
            frame,
            label,
            (int(bbox[0]), int(bbox[1] + 15)),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.6,
            color=color,
            thickness=2,
        )

    return frame


def add_bbox_sequence_to_frame_sequence(frames, double_bboxes):
    T, num_coords = double_bboxes.shape
    assert num_coords == 10
    assert T == len(frames)

    convert_to_torch = False
    if torch.is_tensor(frames):
        frames = frames.numpy()
        convert_to_torch = True

    double_bboxes[double_bboxes == 1000] = 0

    for i, frame in enumerate(frames):
        bbox_list = [double_bboxes[i][:4], double_bboxes[i][5:9]]
        add_bboxes_to_frame(
            frame,
            bbox_list,
            labels=None,
            inplace=True,
            colors=[(255, 0, 0), (0, 255, 0)],
            thinkness=2,
        )
    if convert_to_torch:
        result = torch.Tensor(frames).to(torch.uint8)
    else:
        result = frames
    return result


def add_bbox_sensor_to_image(curr_frame, task_observations, det_sensor_key, which_image):
    task_relevant_object_bbox = task_observations[det_sensor_key]
    (bbox_dim,) = task_relevant_object_bbox.shape
    assert bbox_dim in [5, 10]
    if bbox_dim == 5:
        task_relevant_object_bboxes = [task_relevant_object_bbox[:4]]
    if bbox_dim == 10:
        task_relevant_object_bboxes = [
            task_relevant_object_bbox[:4],
            task_relevant_object_bbox[5:9],
        ]
        task_relevant_object_bboxes = [
            b for b in task_relevant_object_bboxes if b[1] <= curr_frame.shape[0]
        ]
    if which_image == 'nav':
        pass
    elif which_image == 'manip':
        start_index = curr_frame.shape[1] // 2
        for i in range(len(task_relevant_object_bboxes)):
            task_relevant_object_bboxes[i][0] += start_index
            task_relevant_object_bboxes[i][2] += start_index
    else:
        raise NotImplementedError
    if len(task_relevant_object_bboxes) > 0:
        # This works because the navigation frame comes first in curr_frame
        add_bboxes_to_frame(
            frame=curr_frame,
            bboxes=task_relevant_object_bboxes,
            labels=None,
            inplace=True,
        )


def get_top_down_path_view(
    controller: StretchController,
    agent_path: Sequence[Dict[str, float]],
    targets_to_highlight=None,
    orthographic: bool = True,
    map_height_width=(1000, 1000),
    path_width: float = 0.045,
):
    thor_controller = controller.controller

    original_hw = thor_controller.last_event.frame.shape[:2]

    if original_hw != map_height_width:
        event = thor_controller.step(
            'ChangeResolution', x=map_height_width[1], y=map_height_width[0], raise_for_failure=True
        )

    if len(thor_controller.last_event.third_party_camera_frames) < 2:
        event = thor_controller.step('GetMapViewCameraProperties', raise_for_failure=True)
        cam = copy.deepcopy(event.metadata['actionReturn'])
        if not orthographic:
            bounds = event.metadata['sceneBounds']['size']
            max_bound = max(bounds['x'], bounds['z'])

            cam['fieldOfView'] = 50
            cam['position']['y'] += 1.1 * max_bound
            cam['orthographic'] = False
            cam['farClippingPlane'] = 50
            del cam['orthographicSize']

        event = thor_controller.step(
            action='AddThirdPartyCamera',
            **cam,
            skyboxColor='white',
            raise_for_failure=True,
        )

    waypoints = []
    for target in targets_to_highlight or []:
        target_position = controller.get_object_position(target)
        target_dict = {
            'position': target_position,
            'color': {'r': 1, 'g': 0, 'b': 0, 'a': 1},
            'radius': 0.5,
            'text': '',
        }
        waypoints.append(target_dict)

    if len(agent_path) != 0:
        thor_controller.step(
            action='VisualizeWaypoints',
            waypoints=waypoints,
            raise_for_failure=True,
        )
        # put this over the waypoints just in case
        event = thor_controller.step(
            action='VisualizePath',
            positions=agent_path,
            pathWidth=path_width,
            raise_for_failure=True,
        )
        thor_controller.step({'action': 'HideVisualizedPath'})

    map = event.third_party_camera_frames[-1]

    if original_hw != map_height_width:
        thor_controller.step(
            'ChangeResolution', x=original_hw[1], y=original_hw[0], raise_for_failure=True
        )

    return map


def get_top_down_frame(controller, agent_path, target_ids):
    top_down = controller.get_top_down_path_view(agent_path, target_ids)
    return top_down


class VideoLogging:
    @staticmethod
    def get_video_frame(
        agent_frame: np.ndarray,
        frame_number: int,
        action_names: List[str],
        action_dist: Optional[List[float]],
        ep_length: int,
        last_action_success: Optional[bool],
        taken_action: Optional[str],
        task_desc: str,
    ) -> np.array:
        agent_height, agent_width, ch = agent_frame.shape

        font_to_use = 'Arial.ttf'  # possibly need a full path here
        full_font_load = ImageFont.truetype(font_to_use, 14)

        IMAGE_BORDER = 25
        TEXT_OFFSET_H = 60
        TEXT_OFFSET_V = 30

        image_dims = (
            agent_height + 2 * IMAGE_BORDER + 30,
            agent_width + 2 * IMAGE_BORDER + 400,
            ch,
        )
        image = np.full(image_dims, 255, dtype=np.uint8)

        image[
            IMAGE_BORDER : IMAGE_BORDER + agent_height, IMAGE_BORDER : IMAGE_BORDER + agent_width, :
        ] = agent_frame

        text_image = Image.fromarray(image)
        img_draw = ImageDraw.Draw(text_image)
        # font size 25, aligned center and middle
        if action_dist is not None:
            for i, (prob, action) in enumerate(zip(action_dist, action_names)):
                try:
                    action_long_name = stretch_long_names[action]
                except KeyError:
                    action_long_name = action
                if i < 10:
                    img_draw.text(
                        (
                            IMAGE_BORDER * 2 + agent_width + TEXT_OFFSET_H,
                            (TEXT_OFFSET_V + 5) + i * 10,
                        ),
                        action_long_name,
                        font=ImageFont.truetype(font_to_use, 10),
                        fill='gray' if action != taken_action else 'black',
                        anchor='rm',
                    )
                    img_draw.rectangle(
                        (
                            IMAGE_BORDER * 2 + agent_width + (TEXT_OFFSET_H + 5),
                            TEXT_OFFSET_V + i * 10,
                            IMAGE_BORDER * 2 + agent_width + (TEXT_OFFSET_H + 5) + int(100 * prob),
                            (TEXT_OFFSET_V + 5) + i * 10,
                        ),
                        outline='blue',
                        fill='blue',
                    )
                else:
                    img_draw.text(
                        (
                            IMAGE_BORDER * 2 + agent_width + TEXT_OFFSET_H + 200,
                            (TEXT_OFFSET_V + 5) + (i - 10) * 10,
                        ),
                        action_long_name,
                        font=ImageFont.truetype(font_to_use, 10),
                        fill='gray' if action != taken_action else 'black',
                        anchor='rm',
                    )
                    img_draw.rectangle(
                        (
                            IMAGE_BORDER * 2 + agent_width + (TEXT_OFFSET_H + 205),
                            TEXT_OFFSET_V + (i - 10) * 10,
                            IMAGE_BORDER * 2
                            + agent_width
                            + (TEXT_OFFSET_H + 205)
                            + int(100 * prob),
                            (TEXT_OFFSET_V + 5) + (i - 10) * 10,
                        ),
                        outline='blue',
                        fill='blue',
                    )

        img_draw.text(
            (IMAGE_BORDER * 1.1, IMAGE_BORDER * 1),
            str(frame_number),
            font=full_font_load,  # ImageFont.truetype(font_to_use, 25),
            fill='white',
        )

        if last_action_success is not None:
            img_draw.text(
                (IMAGE_BORDER * 2 + agent_width + TEXT_OFFSET_H, IMAGE_BORDER * 1 + 235),
                'Last Action:',
                font=full_font_load,  # ImageFont.truetype(font_to_use, 14),
                fill='gray',
                anchor='rm',
            )
            img_draw.text(
                (IMAGE_BORDER * 2 + agent_width + TEXT_OFFSET_H, IMAGE_BORDER * 1 + 235),
                ' Success' if last_action_success else ' Failure',
                font=full_font_load,  # ImageFont.truetype(font_to_use, 14),
                fill='green' if last_action_success else 'red',
                anchor='lm',
            )

        if taken_action == 'manual override':
            img_draw.text(
                (IMAGE_BORDER * 2 + agent_width + TEXT_OFFSET_H + 50, TEXT_OFFSET_V + 5 * 20),
                'Manual Override',
                font=full_font_load,  # ImageFont.truetype(font_to_use, 14),
                fill='red',
                anchor='rm',
            )

        img_draw.text(
            (IMAGE_BORDER * 2 + agent_width + TEXT_OFFSET_H, IMAGE_BORDER * 1 + 145),
            'Target Dist:',
            font=full_font_load,  # ImageFont.truetype(font_to_use, 14),
            fill='gray',
            anchor='rm',
        )
        img_draw.text(
            (IMAGE_BORDER * 2 + agent_width + TEXT_OFFSET_H, IMAGE_BORDER * 1 + 145),
            f' Task: {task_desc}',
            font=full_font_load,  # ImageFont.truetype(font_to_use, 14),
            fill='gray',
            anchor='lm',
        )

        lower_offset = 10
        progress_bar_height = 20

        img_draw.rectangle(
            (
                IMAGE_BORDER,
                agent_height + IMAGE_BORDER + lower_offset,
                IMAGE_BORDER + agent_width,
                agent_height + IMAGE_BORDER + progress_bar_height + lower_offset,
            ),
            outline='lightgray',
            fill='lightgray',
        )
        img_draw.rectangle(
            (
                IMAGE_BORDER,
                agent_height + IMAGE_BORDER + lower_offset,
                IMAGE_BORDER + int(frame_number * agent_width / ep_length),
                agent_height + IMAGE_BORDER + progress_bar_height + lower_offset,
            ),
            outline='blue',
            fill='blue',
        )

        return np.array(text_image)
