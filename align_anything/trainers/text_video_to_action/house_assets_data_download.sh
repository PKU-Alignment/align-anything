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
export HOME_PREFIX=/your/local/data/path


export OBJAVERSE_HOUSES_BASE_DIR=${HOME_PREFIX}/houses/objaverse_houses
export OBJAVERSE_HOUSES_DIR=${HOME_PREFIX}/houses/objaverse_houses/houses_2023_07_28
export OBJAVERSE_DATA_BASE_DIR=${HOME_PREFIX}/assets/objaverse_houses
export OBJAVERSE_DATA_DIR=${HOME_PREFIX}/assets/objaverse_assets/2023_07_28
export OBJAVERSE_ANNOTATIONS_PATH=${HOME_PREFIX}/assets/objaverse_assets/2023_07_28/annotations.json.gz


echo "Download objaverse assets and annotation"
if [ ! -f $OBJAVERSE_DATA_BASE_DIR/2023_07_28/annotations.json.gz ] ; then
  python -m objathor.dataset.download_annotations --version 2023_07_28 --path $OBJAVERSE_DATA_BASE_DIR
else
  echo "Annotations already downloaded"
fi

if [ ! -d $OBJAVERSE_DATA_BASE_DIR/2023_07_28/assets ] ; then
  python -m objathor.dataset.download_assets --version 2023_07_28 --path $OBJAVERSE_DATA_BASE_DIR
else
  echo "Assets already downloaded"
fi

echo "Download objaverse houses"
if [ ! -f $OBJAVERSE_HOUSES_BASE_DIR/houses_2023_07_28/val.jsonl.gz ] ; then
  python download_objaverse_houses.py --save_dir $OBJAVERSE_HOUSES_BASE_DIR --subset val
else
  echo "Houses already downloaded"
fi
