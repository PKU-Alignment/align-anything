#!/bin/bash
#
# Copyright 2025 Allen Institute for AI

# Copyright 2025 Align-Anything Team. All Rights Reserved.
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

# set the color
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color
PURPLE='\033[0;35m'
BLUE='\033[0;34m'

# set the test directory, default is the current directory
TEST_DIR=${1:-.}
OUTPUT_ROOT_DIR=${2:-../outputs}

# create the output directory if it does not exist
mkdir -p $OUTPUT_ROOT_DIR

export HF_DATASETS_CACHE=$OUTPUT_ROOT_DIR/cache

echo -e "${BLUE}Setting $HF_DATASETS_CACHE as the huggingface cache directory. We will delete it after the test.${NC}"


export OUTPUT_ROOT_DIR=$OUTPUT_ROOT_DIR

# statistics variables
TOTAL=0
FAILED=0

echo -e "${PURPLE}Start testing all scripts in directory ${TEST_DIR}...${NC}"
echo "----------------------------------------"

NUM_SCRIPT=$(find "$TEST_DIR" -name "*.sh" -type f | sort | wc -l)

# find and execute all .sh files
for script in $(find "$TEST_DIR" -name "*.sh" -type f | sort); do
    # skip the current script itself
    if [ "$(basename "$script")" = "$(basename "$0")" ]; then
        continue
    fi

    # increase the total count
    ((TOTAL++))

    echo -e "${YELLOW}Start testing script: ${script} (${TOTAL}/${NUM_SCRIPT}) ${NC}"

    # execute the script and redirect the output to a temporary file
    temp_output=$(mktemp)
    if ! bash "$script" > "$temp_output" 2>&1; then
        # script execution failed
        ((FAILED++))
        echo -e "${RED}✗ Script failed: ${script}${NC}"
        echo -e "${RED}Error information:${NC}"
        cat "$temp_output" | sed 's/^/  /'
        echo "----------------------------------------"

    fi

    # delete the temporary file
    rm -f "$temp_output"
    echo -e "${GREEN}Test script completed: ${script} (${TOTAL}/${NUM_SCRIPT})${NC}"
done

# output the statistics information
echo "----------------------------------------"
if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}All scripts passed! Total executed: $TOTAL scripts${NC}"
else
    echo -e "${RED}Test completed: Total executed $TOTAL scripts, $FAILED failed${NC}"
fi

echo -e "${BLUE}Deleting the output directory...${NC}"
rm -rf $OUTPUT_ROOT_DIR

exit $FAILED
