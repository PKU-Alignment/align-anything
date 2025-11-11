export OBJAVERSE_DATA_DIR="path/to/objaverse_assets/2023_07_28"
set -x

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${SCRIPT_DIR}/../eval_anything" || exit 1

python -m benchmarks.text_vision_to_action.chores.eval
