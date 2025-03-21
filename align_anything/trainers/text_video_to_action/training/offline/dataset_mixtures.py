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
import sys


def get_mixture_by_name(name):
    return getattr(sys.modules[__name__], name, [name])


CHORES = [
    # "ObjectNavType",
    'PickupType',
    # "FetchType",
    # "RoomVisit",  # "SimpleExploreHouse",  #
]

CHORESNAV = [
    'ObjectNavType',
    'ObjectNavRoom',
    'ObjectNavRelAttribute',
    'ObjectNavAffordance',
    'ObjectNavLocalRef',
    'ObjectNavDescription',
    'RoomNav',
]
