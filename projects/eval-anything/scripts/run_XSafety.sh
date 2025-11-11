set -x

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${SCRIPT_DIR}/../eval_anything" || exit 1

# Optional: set environment variables for API-based model configuration
export API_KEY="sk-xxxxxxxxxxxxxxxxx"
export API_BASE="xxxxxxxxxxxxxxxxx"
export NUM_WORKERS=32   # The number of workers for parallel API inference
export HF_TOKEN="xxxxxxxxxxxxxxxxx"

python __main__.py \
    --eval_info evaluate_XSafety.yaml
