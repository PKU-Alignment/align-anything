# Copyright 2024 Allen Institute for AI
# ==============================================================================

import random

import ai2thor.controller
import torch
import torchvision.transforms
from torchvision.transforms import Compose, Normalize

from eval_anything.third_party.SPOC.utils.constants.stretch_initialization_utils import (
    STRETCH_ENV_ARGS,
)
from eval_anything.third_party.SPOC.utils.data_generation_utils.mp4_utils import save_frames_to_mp4


def get_full_transformation_list(size, version='v2'):
    if version == 'v2':
        return [
            torchvision.transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.2, hue=0.05
            ),
            torchvision.transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 2)),
            torchvision.transforms.RandomResizedCrop(
                size,
                scale=(0.9, 1),
            ),
            torchvision.transforms.RandomPosterize(bits=7, p=0.2),
            torchvision.transforms.RandomPosterize(bits=6, p=0.2),
            torchvision.transforms.RandomPosterize(bits=5, p=0.2),
            torchvision.transforms.RandomPosterize(bits=4, p=0.2),
            torchvision.transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
        ]
    elif version == 'v1':
        return [
            torchvision.transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2
            ),
            torchvision.transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 2)),
            torchvision.transforms.RandomResizedCrop(
                size,
                scale=(0.9, 1),
            ),
            torchvision.transforms.RandomPosterize(bits=7, p=0.3),
            torchvision.transforms.RandomAdjustSharpness(sharpness_factor=2),
            torchvision.transforms.RandomGrayscale(0.2),
        ]
    else:
        raise NotImplementedError(
            f'data augmentation versions supported are v1 and v2, got {version}'
        )


def get_transformation(size=(224, 384)):
    list_of_transformations = get_full_transformation_list(size)
    return Compose(list_of_transformations)


def sample_a_specific_transform(transformation_list, size=(224, 384)):
    specific_transformation = []
    for transform in transformation_list.transforms:

        def sample_value_in_range(list_of_range):
            assert len(list_of_range) == 2
            random_value = random.uniform(list_of_range[0], list_of_range[1])
            return random_value, random_value

        def sample_singular_value(prob):
            return int(random.random() < prob)

        if type(transform) == torchvision.transforms.ColorJitter:
            sampled_brightness = sample_value_in_range(transform.brightness)
            sampled_saturation = sample_value_in_range(transform.saturation)
            sampled_hue = sample_value_in_range(transform.hue)
            sampled_contrast = sample_value_in_range(transform.contrast)

            specific_transformation.append(
                torchvision.transforms.ColorJitter(
                    brightness=sampled_brightness,
                    contrast=sampled_contrast,
                    saturation=sampled_saturation,
                    hue=sampled_hue,
                )
            )
        elif type(transform) == torchvision.transforms.GaussianBlur:
            sampled_sigma = sample_value_in_range(transform.sigma)

            specific_transformation.append(
                torchvision.transforms.GaussianBlur(
                    kernel_size=transform.kernel_size, sigma=sampled_sigma
                )
            )
        elif type(transform) == torchvision.transforms.RandomResizedCrop:
            sampled_scale = sample_value_in_range(transform.scale)

            specific_transformation.append(
                torchvision.transforms.RandomResizedCrop(
                    size,
                    scale=sampled_scale,
                )
            )

        elif type(transform) == torchvision.transforms.RandomPosterize:
            sampled_p = sample_singular_value(transform.p)

            specific_transformation.append(
                torchvision.transforms.RandomPosterize(bits=7, p=sampled_p)
            )
        elif type(transform) == torchvision.transforms.RandomAdjustSharpness:
            sampled_p = sample_singular_value(transform.p)

            specific_transformation.append(
                torchvision.transforms.RandomAdjustSharpness(sharpness_factor=2, p=sampled_p)
            )
        elif type(transform) == torchvision.transforms.RandomGrayscale:
            sampled_p = sample_singular_value(transform.p)

            specific_transformation.append(torchvision.transforms.RandomGrayscale(p=sampled_p))
        elif type(transform) in [torchvision.transforms.Lambda or Normalize]:
            specific_transformation.append(transform)
        else:
            raise NotImplementedError

    return Compose(specific_transformation)


def frame_by_frame_augmentation():
    controller = ai2thor.controller.Controller(**STRETCH_ENV_ARGS)
    action_list = ['MoveAhead', 'RotateRight', 'RotateLeft', 'MoveBack']
    frames = []
    augmented_frames = []
    transformation = get_transformation()
    for i in range(100):
        controller.step(random.choice(action_list))
        frame = controller.last_event.frame
        frames.append(frame)
        frame_tensorized = (
            torch.Tensor(frame.copy()).unsqueeze(0).permute(0, 3, 1, 2).type(torch.uint8)
        )
        augmented_frame = transformation(frame_tensorized).permute(0, 2, 3, 1).squeeze(0).numpy()
        augmented_frames.append(augmented_frame)

    save_frames_to_mp4(
        frames,
        'og_frames.mp4',
        5,
    )
    save_frames_to_mp4(
        augmented_frames,
        'augmented_frames.mp4',
        5,
    )


def test_apply_same_transformation():
    controller = ai2thor.controller.Controller(**STRETCH_ENV_ARGS)
    action_list = ['MoveAhead', 'RotateRight', 'RotateLeft', 'MoveBack']
    frames = []
    augmented_frames = []
    transformation = get_transformation()
    transformation = sample_a_specific_transform(transformation)

    for i in range(100):
        controller.step(random.choice(action_list))
        frame = controller.last_event.frame
        frames.append(frame)
        frame_tensorized = (
            torch.Tensor(frame.copy()).unsqueeze(0).permute(0, 3, 1, 2).type(torch.uint8)
        )
        augmented_frame = transformation(frame_tensorized).permute(0, 2, 3, 1).squeeze(0).numpy()
        augmented_frames.append(augmented_frame)

    save_frames_to_mp4(
        frames,
        'og_frames.mp4',
        5,
    )
    save_frames_to_mp4(
        augmented_frames,
        'augmented_frames.mp4',
        5,
    )
