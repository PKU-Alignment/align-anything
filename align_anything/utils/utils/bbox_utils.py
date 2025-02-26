from typing import Tuple

import numpy as np
import torch

BBOX_DIST_THRESHOLD = 0.1


def get_box_from_object(obj, verbose=False):
    if obj.get("objectOrientedBoundingBox") is not None:
        box = obj["objectOrientedBoundingBox"]["cornerPoints"]
    else:
        if verbose:
            print(f"Using axisAlignedBoundingBox for {obj['objectId']} ({obj['synset']})")
        box = obj["axisAlignedBoundingBox"]["cornerPoints"]

    return np.array(box)


# lightly adapted from https://github.com/allenai/ai2thor-rearrangement
def get_basis_for_3d_box_from_bbox_corners(bbox_corners) -> Tuple[np.ndarray, np.ndarray]:
    # stacked along columns
    without_first = first_corner_to_other_vertices_vectors(bbox_corners)
    magnitudes1 = vector_lengths(without_first)
    v0_ind = np.argmin(magnitudes1)
    v0_mag = magnitudes1[v0_ind]

    if v0_mag < 1e-8:
        raise RuntimeError(f"Could not find basis for {bbox_corners}")

    v0 = without_first[np.argmin(magnitudes1)] / v0_mag

    orth_to_v0 = (v0.reshape(1, -1) * without_first).sum(-1) < v0_mag / 2.0
    inds_orth_to_v0 = np.where(orth_to_v0)[0]
    v1_ind = inds_orth_to_v0[np.argmin(magnitudes1[inds_orth_to_v0])]
    v1_mag = magnitudes1[v1_ind]
    v1 = without_first[v1_ind, :] / magnitudes1[v1_ind]

    orth_to_v1 = (v1.reshape(1, -1) * without_first).sum(-1) < v1_mag / 2.0
    inds_orth_to_v0_and_v1 = np.where(orth_to_v0 & orth_to_v1)[0]

    if len(inds_orth_to_v0_and_v1) != 1:
        raise RuntimeError(f"Could not find basis for {bbox_corners}")

    v2_ind = inds_orth_to_v0_and_v1[0]
    v2 = without_first[v2_ind, :] / magnitudes1[v2_ind]

    orth_mat = np.stack((v0, v1, v2), axis=1)  # Orthonormal matrix, stacked by columns!

    return orth_mat, magnitudes1[[v0_ind, v1_ind, v2_ind]]


def get_basis_for_3d_box(obj) -> Tuple[np.ndarray, np.ndarray]:
    # should get object aligned but might return axis aligned
    bbox_corners = get_box_from_object(obj, verbose=False)

    return get_basis_for_3d_box_from_bbox_corners(bbox_corners)


def first_corner_to_other_vertices_vectors(box):
    corner_to_others = box - box[:1, :]
    assert np.isclose(corner_to_others[0].sum(), 0.0)
    without_corner = corner_to_others[1:]
    return without_corner


def vector_lengths(vectors):
    return np.sqrt((vectors * vectors).sum(1))


def get_best_of_two_bboxes(bbox_1, bbox_2):
    assert bbox_1.shape == bbox_2.shape
    B, T, dim = bbox_1.shape
    assert dim == 10
    size_target_obj_box_1 = bbox_1[:, :, 4]
    size_target_obj_box_2 = bbox_2[:, :, 4]

    box_2_is_bigger = size_target_obj_box_1 < size_target_obj_box_2
    bigger_box_obj = bbox_1.clone() if torch.is_tensor(bbox_1) else np.copy(bbox_1)
    bigger_box_obj[box_2_is_bigger] = bbox_2[box_2_is_bigger]

    size_receptacle_box_1 = bbox_1[:, :, 9]
    size_receptacle_box_2 = bbox_2[:, :, 9]

    box_2_is_bigger = size_receptacle_box_1 < size_receptacle_box_2
    bigger_box_rec = bbox_1.clone() if torch.is_tensor(bbox_1) else np.copy(bbox_1)
    bigger_box_rec[box_2_is_bigger] = bbox_2[box_2_is_bigger]

    bigger_box_obj[:, :, 5:9] = bigger_box_rec[:, :, 5:9]
    return bigger_box_obj
