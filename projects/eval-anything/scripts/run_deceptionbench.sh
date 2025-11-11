set -x

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${SCRIPT_DIR}/../eval_anything" || exit 1

# Optional: set environment variables for API-based model configuration
export API_KEY="sk-xxxxx"
export API_BASE="xxxxxxxxxxxxxx"
export NUM_WORKERS=32   # The number of workers for parallel API inference

python __main__.py \
    --eval_info evaluate_deceptionbench.yaml
