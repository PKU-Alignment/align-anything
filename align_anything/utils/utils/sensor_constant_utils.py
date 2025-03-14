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


class AbstractSensor:
    sensor_uuid = 'sensor_name'


class ObservationSensor(AbstractSensor):
    pass


class ManipulationCamera(ObservationSensor):
    sensor_uuid = 'raw_manipulation_camera'


class NavigationCamera(ObservationSensor):
    sensor_uuid = 'raw_navigation_camera'


class NavigationCameraDuplicate(ObservationSensor):
    sensor_uuid = 'raw_navigation_camera_2'


class ManipulationCameraDuplicate(ObservationSensor):
    sensor_uuid = 'raw_manipulation_camera_2'


def is_a_visual_sensor(sensor):
    return sensor in [
        ManipulationCamera.sensor_uuid,
        NavigationCamera.sensor_uuid,
        NavigationCameraDuplicate.sensor_uuid,
        ManipulationCameraDuplicate.sensor_uuid,
    ]  # more can be added later


def is_a_non_visual_sensor(sensor):
    return sensor in [
        'relative_arm_location_metadata',
        'an_object_is_in_hand',
        'last_actions',
        'rooms_seen',
        'room_current_seen',
        'rooms_seen_output',
        'room_current_seen_output',
        'nav_task_relevant_object_bbox',
        'manip_task_relevant_object_bbox',
        'nav_accurate_object_bbox',
        'manip_accurate_object_bbox',
    ]
