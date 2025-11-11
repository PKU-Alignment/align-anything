# Copyright 2024 Allen Institute for AI
# ==============================================================================

import copy
from typing import List, Optional, Sequence

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from eval_anything.third_party.SPOC.utils.constants.stretch_initialization_utils import (
    stretch_long_names,
)


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


# TODO: plan1 save the path with unsafe points
def get_top_down_frame(controller, agent_path, target_ids):
    top_down, agent_path = controller.get_top_down_path_view(agent_path, target_ids)
    return top_down, agent_path


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
        debug: Optional[any],
        # cost: Optional[float],
        # sum_danger : Optional[float],
        # sum_corner : Optional[float],
        # sum_blind : Optional[float],
        # camera_seen: Optional[List[str]],
        # last_objects_causing_cost_list: Optional[List[str]],
        # ignore_objects_name: Optional[List[str]],
        # clusters : Optional[List[List[str]]],
        # critical_objects : Optional[List[str]],
        # danger_equipment : Optional[List[str]],
        # cost_objects_name: Optional[List[str]],
    ) -> np.array:
        agent_height, agent_width, ch = agent_frame.shape

        font_to_use = 'Arial.ttf'  # possibly need a full path here
        full_font_load = ImageFont.truetype(font_to_use, 14)

        IMAGE_BORDER = 25
        TEXT_OFFSET_H = 60
        TEXT_OFFSET_V = 30

        image_dims = (
            agent_height + 2 * IMAGE_BORDER + 30,
            agent_width + 2 * IMAGE_BORDER + 1000,
            ch,
        )
        image = np.full(image_dims, 255, dtype=np.uint8)
        image[
            IMAGE_BORDER : IMAGE_BORDER + agent_height, IMAGE_BORDER : IMAGE_BORDER + agent_width, :
        ] = agent_frame

        text_image = Image.fromarray(image)
        img_draw = ImageDraw.Draw(text_image)
        sum_cost = debug.get('sum_cost', None)
        sum_danger = debug.get('sum_danger', None)
        sum_corner = debug.get('sum_corner', None)
        sum_blind = debug.get('sum_blind', None)
        sum_critical = debug.get('sum_critical', None)
        sum_fragile = debug.get('sum_fragile', None)
        fragile_objects = debug.get('fragile_objects', None)
        camera_seen = debug.get('camera_seen', None)  # 相机可见物体
        last_objects_causing_cost_list = debug.get(
            'last_objects_causing_cost_list', None
        )  # 上一步触发成本的物体
        clusters = debug.get('clusters', None)  # 物体聚类
        critical_objects = debug.get('critical_objects', None)  # 不稳定物体
        danger_objects = debug.get('danger_objects', None)  # 危险物体
        ignore_objects_name = debug.get('ignore_objects_name', None)  # 忽略物体
        error_message = debug.get('error_message', None)  # 错误信息
        sum_robot = debug.get('sum_robot', None)
        sum_object = debug.get('sum_object', None)
        filtered_clusters = debug.get('filtered_clusters', None)  # 物体聚类

        # font size 25, aligned center and middle
        # if cost_objects_name is not None:
        #     img_draw.text(
        #             (IMAGE_BORDER * 2 + agent_width + (TEXT_OFFSET_H + 100), IMAGE_BORDER * 1 + (TEXT_OFFSET_V - 50)),
        #             f"{len(cost_objects_name)}",
        #             font=full_font_load,
        #             fill="red",
        #             anchor="lm",
        #         )
        #     for i, obj in enumerate(cost_objects_name):
        #         img_draw.text(
        #             (IMAGE_BORDER * 2 + agent_width + (TEXT_OFFSET_H + 100), IMAGE_BORDER * 1 + (TEXT_OFFSET_V - 30) + i * 10),
        #             f"{obj}",
        #             font=full_font_load,
        #             fill="red",
        #             anchor="lm",
        #         )
        # 1. 相机可见物体
        if camera_seen is not None:
            img_draw.text(
                (IMAGE_BORDER * 2 + agent_width + (TEXT_OFFSET_H + 180), IMAGE_BORDER * 1 + 105),
                # f"Ignore objects(Unity Bug)",
                f'Camera Seen Objects',
                font=full_font_load,
                fill='black',
                anchor='lm',
            )
            for i, obj in enumerate(camera_seen):
                flag = False
                for obj_d in danger_objects:
                    if obj_d.lower() in obj.lower():
                        flag = True
                img_draw.text(
                    (
                        IMAGE_BORDER * 2 + agent_width + (TEXT_OFFSET_H + 180),
                        IMAGE_BORDER * 1 + 120 + i * 13,
                    ),
                    f'{obj}',
                    font=full_font_load,
                    fill='red' if flag else 'gray',
                    anchor='lm',
                )
        # 2. 动作分布
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

        # 3. 目标任务
        img_draw.text(
            (IMAGE_BORDER * 2 + agent_width, IMAGE_BORDER * 1 + 90),
            f' Task: {task_desc}',
            font=full_font_load,  # ImageFont.truetype(font_to_use, 14),
            fill='gray',
            anchor='lm',
        )
        # 4. 帧数
        img_draw.text(
            (IMAGE_BORDER * 1.1, IMAGE_BORDER * 1),
            str(frame_number),
            font=full_font_load,  # ImageFont.truetype(font_to_use, 25),
            fill='white',
        )
        # 5. 上一步动作成功与否
        if last_action_success is not None:
            img_draw.text(
                (IMAGE_BORDER * 2 + agent_width + TEXT_OFFSET_H, IMAGE_BORDER * 1 + 110),
                'Last Action:',
                font=full_font_load,  # ImageFont.truetype(font_to_use, 14),
                fill='gray',
                anchor='rm',
            )
            img_draw.text(
                (IMAGE_BORDER * 2 + agent_width + TEXT_OFFSET_H, IMAGE_BORDER * 1 + 110),
                ' Success' if last_action_success else ' Failure',
                font=full_font_load,  # ImageFont.truetype(font_to_use, 14),
                fill='green' if last_action_success else 'red',
                anchor='lm',
            )
        # 6. 错误信息

        if error_message is not None and error_message != '':
            split_char = '\''
            error_obj = None
            try:
                error_obj = error_message.split(split_char)[1]
            except:
                print('===vis error', error_message)
            error_obj = error_message
            img_draw.text(
                (IMAGE_BORDER * 2 + agent_width + TEXT_OFFSET_H - 20, IMAGE_BORDER * 1 + 130),
                f'Error: {error_obj}',
                font=full_font_load,  # ImageFont.truetype(font_to_use, 14),
                fill='red',
                anchor='lm',
            )
        # 7. 5个子成本成本
        if sum_robot is not None:
            img_draw.text(
                (IMAGE_BORDER * 2 + agent_width + TEXT_OFFSET_H - 20, IMAGE_BORDER * 1 + 145),
                f'Robot Cost: {sum_robot}',
                font=full_font_load,
                fill='red',
                anchor='lm',
            )
        if sum_object is not None:
            img_draw.text(
                (IMAGE_BORDER * 2 + agent_width + TEXT_OFFSET_H - 20, IMAGE_BORDER * 1 + 160),
                f'Object Cost: {sum_object}',
                font=full_font_load,
                fill='red',
                anchor='lm',
            )

        if sum_cost is not None:
            img_draw.text(
                (IMAGE_BORDER * 2 + agent_width + TEXT_OFFSET_H - 20, IMAGE_BORDER * 1 + 175),
                f'Total Cost: {sum_cost}',
                font=full_font_load,
                fill='red',
                anchor='lm',
            )
        if sum_corner is not None:
            img_draw.text(
                (IMAGE_BORDER * 2 + agent_width + TEXT_OFFSET_H - 20, IMAGE_BORDER * 1 + 190),
                f'Corner Cost: {sum_corner}',
                font=full_font_load,
                fill='red',
                anchor='lm',
            )
        if sum_blind is not None:
            img_draw.text(
                (IMAGE_BORDER * 2 + agent_width + TEXT_OFFSET_H - 20, IMAGE_BORDER * 1 + 205),
                f'Blind Spot Cost: {sum_blind}',
                font=full_font_load,
                fill='red',
                anchor='lm',
            )
        if sum_danger is not None:
            img_draw.text(
                (IMAGE_BORDER * 2 + agent_width + TEXT_OFFSET_H - 20, IMAGE_BORDER * 1 + 220),
                f"Danger Cost: {sum_danger} {danger_objects if len(danger_objects) > 0 else ''}",
                font=full_font_load,
                fill='red',
                anchor='lm',
            )
            # img_draw.text(
            #     (IMAGE_BORDER * 2 + agent_width + TEXT_OFFSET_H - 20, IMAGE_BORDER * 1 + 205),
            #     f"{danger_objects}",
            #     font=full_font_load,
            #     fill="red",
            #     anchor="lm",
            # )
        if sum_fragile is not None:
            img_draw.text(
                (IMAGE_BORDER * 2 + agent_width + TEXT_OFFSET_H - 20, IMAGE_BORDER * 1 + 235),
                f'Fragile Cost: {sum_fragile}',
                font=full_font_load,
                fill='red',
                anchor='lm',
            )
            # img_draw.text(
            #     (IMAGE_BORDER * 2 + agent_width + TEXT_OFFSET_H - 20, IMAGE_BORDER * 1 + 235),
            #     f"{fragile_objects}",
            #     font=full_font_load,
            #     fill="red",
            #     anchor="lm",
            # )
        if sum_critical is not None:
            img_draw.text(
                (IMAGE_BORDER * 2 + agent_width + TEXT_OFFSET_H - 20, IMAGE_BORDER * 1 + 250),
                f"Critical Cost: {sum_critical} {critical_objects if len(critical_objects) > 0 else ''}",
                font=full_font_load,
                fill='red',
                anchor='lm',
            )
            # img_draw.text(
            #     (IMAGE_BORDER * 2 + agent_width + TEXT_OFFSET_H - 20, IMAGE_BORDER * 1 + 265),
            #     f"{critical_objects}",
            #     font=full_font_load,
            #     fill="red",
            #     anchor="lm",
            # )

        # 7. 聚类
        if last_objects_causing_cost_list is not None:
            img_draw.text(
                (IMAGE_BORDER * 2 + agent_width + TEXT_OFFSET_H + 400, IMAGE_BORDER * 1),
                f'Fragile Objects',
                font=full_font_load,
                fill='black',
                anchor='lm',
            )
            if fragile_objects is not None:
                for count, obj in enumerate(fragile_objects):
                    img_draw.text(
                        (
                            IMAGE_BORDER * 2 + agent_width + TEXT_OFFSET_H + 400,
                            IMAGE_BORDER * 1 + 15 + count * 15,
                        ),
                        f"{obj['name']}",
                        font=full_font_load,
                        fill='black',
                        anchor='lm',
                    )
        # 8. 忽略物体
        if ignore_objects_name is not None:
            img_draw.text(
                (IMAGE_BORDER * 2 + agent_width + (TEXT_OFFSET_H + 580), IMAGE_BORDER * 1),
                f'Ignore Objects',
                font=full_font_load,
                fill='black',
                anchor='lm',
            )
            for i, obj in enumerate(ignore_objects_name):
                img_draw.text(
                    (
                        IMAGE_BORDER * 2 + agent_width + (TEXT_OFFSET_H + 580),
                        IMAGE_BORDER * 1 + 13 + i * 13,
                    ),
                    f'{obj}',
                    font=full_font_load,
                    fill='gray',
                    anchor='lm',
                )
        # 9. 触发成本的物体
        if ignore_objects_name is not None:
            img_draw.text(
                (IMAGE_BORDER * 2 + agent_width + (TEXT_OFFSET_H + 720), IMAGE_BORDER * 1),
                f'Cost Objects',
                font=full_font_load,
                fill='black',
                anchor='lm',
            )
            for i, obj in enumerate(last_objects_causing_cost_list):
                img_draw.text(
                    (
                        IMAGE_BORDER * 2 + agent_width + (TEXT_OFFSET_H + 720),
                        IMAGE_BORDER * 1 + 13 + i * 13,
                    ),
                    f'{obj[0]} {obj[2]}',
                    font=full_font_load,
                    fill='gray',
                    anchor='lm',
                )
        # if taken_action == "manual override":
        #     img_draw.text(
        #         (IMAGE_BORDER * 2 + agent_width + TEXT_OFFSET_H + 50, TEXT_OFFSET_V + 5 * 20),
        #         "Manual Override",
        #         font=full_font_load,  # ImageFont.truetype(font_to_use, 14),
        #         fill="red",
        #         anchor="rm",
        #     )

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
