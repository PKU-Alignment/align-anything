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
import math
from typing import Optional, List, Sequence, Dict
from typing import TYPE_CHECKING

import numpy as np
from shapely.geometry import Point, Polygon, GeometryCollection
from shapely.ops import triangulate
from skimage.morphology import skeletonize

from align_anything.utils.utils.bbox_utils import get_basis_for_3d_box
from align_anything.utils.utils.data_generation_utils.loc_grid_conversion import locs2grids, grids2locs

if TYPE_CHECKING:
    from environment.stretch_controller import StretchController


ALIGNMENT_THRESHOLD = 10  # degrees allowed between agent heading and object center
PROP_VISIBLE_THRESHOLD = 0.8


def vector_dif(loc_start, loc_goal):
    cur_x = loc_start["x"]
    cur_z = loc_start["z"]
    goal_x = loc_goal["x"]
    goal_z = loc_goal["z"]
    vector = (goal_x - cur_x, goal_z - cur_z)
    return vector


def rotation_from(full_agent_pose, goal_obj_position):
    cur_heading = full_agent_pose["rotation"]["y"]
    vector = vector_dif(full_agent_pose["position"], goal_obj_position)
    if vector[1] == 0 and vector[0] == 0:
        result = cur_heading
    else:
        result = math.degrees(math.atan2(vector[0], (vector[1])))

    result = result - cur_heading
    result %= 360
    if result > 180:
        result = result - 360
    return result


def get_room_id_from_location(room_polymap, position, verbose=True):
    if type(position) == dict and "x" in position and "z" in position:
        point = Point(position["x"], position["z"])
    else:
        assert len(position) == 3
        point = Point(position[0], position[2])

    room_id_to_dist = {}
    for room_id, poly in room_polymap.items():
        room_id_to_dist[room_id] = poly.distance(point)
        if room_id_to_dist[room_id] == 0:
            return room_id

    on_walls_of_room_ids = [room_id for room_id, dist in room_id_to_dist.items() if dist < 1e-3]
    if len(on_walls_of_room_ids) == 0:
        if verbose:
            print(position, "is out of house in get_room_id_from_location")
        return None
    elif len(on_walls_of_room_ids) == 1:
        return on_walls_of_room_ids[0]
    else:
        if verbose:
            print(position, "is on walls of multiple rooms, cannot return unique room id")
        return None


def get_rooms_polymap_and_type(house):
    room_poly_map = {}
    room_type_dict = {}
    # NOTE: Map the rooms
    for i, room in enumerate(house["rooms"]):
        room_poly_map[room["id"]] = Polygon([(p["x"], p["z"]) for p in room["floorPolygon"]])
        room_type_dict[room["id"]] = room["roomType"]
    return room_poly_map, room_type_dict


def thinned_starting_positions(locations, grid_spacing=0.25):
    im, locs = locs2grids(locations, grid_spacing)

    im2 = skeletonize(im)
    num_pos = np.sum(im2)

    if num_pos == 0:
        # trying with all locations
        return locations

    return grids2locs(im2, locs)


def get_wall_center_floor_level(wall_id, y):
    wall_xzs = wall_id.split("|")[2:]
    assert len(wall_xzs) == 4

    return dict(
        x=(float(wall_xzs[0]) + float(wall_xzs[2])) / 2,
        y=y,
        z=(float(wall_xzs[1]) + float(wall_xzs[3])) / 2,
    )


def is_any_object_sufficiently_visible_and_in_center_frame(
    controller: "StretchController",
    object_ids: List[str],
    scale: float = 1.5e4,
    manipulation_camera=False,
    absolute_min_pixels: int = 200,
):
    # Note: this code assumes INTEL_VERTICAL_FOV = 59 is used
    # Note: small object of e.g. 0.15 * 0.15 -> 0.0225 m2 (scale ~10,000 for 200 pixels)
    # Also, constraining the maximum threshold to be 1000 (the old one for non-SMALL_TYPES),
    # and the minimum to be 200 (the old one for SMALL_TYPES).
    # Note: Using default scale=15,000 instead of 10,000 as `ProportionOfObjectVisible` resolves it.
    # Note: default image height = 224 -> 224 ^ 2 = 50176 pixels is the area for a 1:1 image shape ratio.
    scale_to_apply = scale * (controller.navigation_camera.shape[0] ** 2) / 50176
    pixel_mass_thresholds = {}
    for oid in object_ids:
        if manipulation_camera:
            pixel_mass_thresholds[oid] = 200
        else:
            # # d1, d2, d3 are the dimensions of the bbox, in meters
            try:
                _, mags = get_basis_for_3d_box(controller.get_object(oid))
            except (KeyboardInterrupt, SystemExit):
                raise
            except:
                # This is a hack to deal with the fact that the basis `get_basis_for_3d_box` may
                # fail for some objects. In that case, we just use the default value of 200.
                pixel_mass_thresholds[oid] = 200
                continue
            d1, d2, d3 = tuple(mags)
            largest_bbox_face_area = max(d1 * d2, d2 * d3, d3 * d1)
            pixel_mass_thresholds[oid] = max(
                min(scale_to_apply * largest_bbox_face_area, 1000), absolute_min_pixels
            )

    oid_to_seg = None
    target_visibility_quant = {}
    for oid in object_ids:
        alignment = abs(
            controller.get_agent_alignment_to_object(oid, use_arm_orientation=manipulation_camera)
        )
        if alignment <= ALIGNMENT_THRESHOLD:
            if oid_to_seg is None:
                # Have to do this here to avoid loading the segmentation masks
                # when we don't need to
                oid_to_seg = (
                    controller.manipulation_camera_segmentation
                    if manipulation_camera
                    else controller.navigation_camera_segmentation
                )
            try:
                pixel_mass = oid_to_seg[oid].sum()
                empty_top = (
                    None  # MANIP CAMERA ONLY: Make sure there are at least 10% empty pixels on top of the image
                    if not manipulation_camera
                    else (
                        oid_to_seg[oid][: int(0.1 * controller.navigation_camera.shape[0])] == 0
                    ).all()
                )
            except KeyError:
                pixel_mass = 0
                empty_top = False

            target_visibility_quant[oid] = {
                "alignment": alignment,
                "pixel_mass": pixel_mass,
                "empty_top": empty_top,
            }

    sufficient_visibility = False
    for obj_id, obj_data in target_visibility_quant.items():
        if obj_data["alignment"] >= ALIGNMENT_THRESHOLD:
            continue

        if obj_data["pixel_mass"] < absolute_min_pixels:
            continue

        if obj_data["pixel_mass"] <= pixel_mass_thresholds[obj_id]:
            # The pixel mass threshold can be overly strict so we include here an approximate check
            # to see if the object is > prop_visible_threshold visible and, if it is, we say its ok.
            prop_visible = controller.step(
                action="ProportionOfObjectVisible",
                objectId=list(target_visibility_quant.keys())[0],
            ).metadata["actionReturn"]
            if prop_visible < PROP_VISIBLE_THRESHOLD:
                continue

        if obj_data["empty_top"] is not None and (not obj_data["empty_top"]):
            continue

        sufficient_visibility = True
        break

    return sufficient_visibility


def triangulate_room_polygon(room_polygon: Polygon) -> GeometryCollection:
    return triangulate(room_polygon)


def snap_to_skeleton(
    controller: "StretchController",
    corners: Sequence[Dict[str, float]],
    thinned_locs: Optional[Sequence[Dict[str, float]]] = None,
    dist_threshold: float = 0.25,
) -> Sequence[Dict[str, float]]:
    """In place modification of corners"""

    if len(corners) > 2:
        # Here we try, one last time, to adjust the path to be further from
        # obstacles on the navmesh. We do this because we might have selected
        # the smallest agent capsule above because there is a single point along
        # the path where the agent barely fits. This results in the agent sometimes
        # walking super close to doors and other obstacles along the ENTIRE path which
        # can cause the agent to get stuck for extended periods of time.
        if thinned_locs is None:
            thinned_locs = thinned_starting_positions(controller.get_reachable_positions())

        thinned_pts = np.array([[p["x"], p["z"]] for p in thinned_locs])

        for i, corner in list(enumerate(corners))[1:-1]:
            p = np.array([[corner["x"], corner["z"]]])
            dists = np.linalg.norm(p - thinned_pts, axis=1)
            if dists.min() <= dist_threshold:
                closest_thinned = thinned_pts[dists.argmin()]
                corner["x"] = float(closest_thinned[0])
                corner["z"] = float(closest_thinned[1])

    return corners
