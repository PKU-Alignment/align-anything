# Example

To facilitate users in assessing the model's performance and effectiveness on multimodal benchmark, we offer a series of evaluation scripts located in the `./scripts` directory. These scripts support various evaluation benchmarks, allowing users to flexibly choose suitable settings to ensure the accuracy and comparability of results. By properly configuring evaluation parameters, users can effectively test and compare the model's performance across various tasks, thereby advancing the model's development and optimization.

### evaluate.sh
The `./scripts/evaluate.sh` script runs model evaluation on the benchmarks, and parameters that require user input have been left empty. The corresponding script is as follow:
~~~sh
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${SCRIPT_DIR}/../align_anything/evaluation" || exit 1

BENCHMARKS=("")
OUTPUT_DIR=""
GENERATION_BACKEND=""
MODEL_ID=""
MODEL_NAME_OR_PATH=""
CHAT_TEMPLATE=""

for BENCHMARK in "${BENCHMARKS[@]}"; do
    python __main__.py \
        --benchmark ${BENCHMARK} \
        --output_dir ${OUTPUT_DIR} \
        --generation_backend ${GENERATION_BACKEND} \
        --model_id ${MODEL_ID} \
        --model_name_or_path ${MODEL_NAME_OR_PATH} \
        --chat_template ${CHAT_TEMPLATE}
done
~~~
- `BENCHMARK`: A list of evaluation benchmarks used to assess the model's performance.
- `OUTPUT_DIR`: The directory for saving the evaluation results and output files.
- `GENERATION_BACKEND`: The backend used for generation, vLLM or deepspeed.
- `MODEL_ID`: Unique identifier for the model, used to track and differentiate between evaluations.
- `MODEL_NAME_OR_PATH`: Local path or Hugging Face link to the model weights.
- `CHAT_TEMPLATE`: The chat template corresponding to the model.

### models_pk.sh

The `./scripts/models_pk.sh` evaluates and compares two models across benchmarks. Ensure all parameters are set before running. The corresponding script is as follow:
~~~sh
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${SCRIPT_DIR}/../align_anything/evaluation" || exit 1

BENCHMARKS=("")
OUTPUT_DIR=""
GENERATION_BACKEND=""
MODEL_IDS=("" "")
MODEL_NAME_OR_PATHS=("" "")
CHAT_TEMPLATES=("" "")

for BENCHMARK in "${BENCHMARKS[@]}"; do
    echo "Processing benchmark: ${BENCHMARK}"
    
    for i in "${!MODEL_IDS[@]}"; do
        MODEL_ID=${MODEL_IDS[$i]}
        MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATHS[$i]}
        CHAT_TEMPLATE=${CHAT_TEMPLATES[$i]}
        
        echo "Running model ${MODEL_ID} for benchmark ${BENCHMARK}"
        python __main__.py \
            --benchmark ${BENCHMARK} \
            --output_dir ${OUTPUT_DIR} \
            --generation_backend ${GENERATION_BACKEND} \
            --model_id ${MODEL_ID} \
            --model_name_or_path ${MODEL_NAME_OR_PATH} \
            --chat_template ${CHAT_TEMPLATE}
    done
    
    python models_pk.py --benchmark ${BENCHMARK} \
                        --model_1 "${MODEL_IDS[0]}" \
                        --model_2 "${MODEL_IDS[1]}"
done

~~~
- `BENCHMARKS`: A list of evaluation benchmarks used to assess the model's performance.
- `OUTPUT_DIR`: The directory for saving the evaluation results and output files.
- `GENERATION_BACKEND`: The backend used for generation, vLLM or deepspeed.
- `MODEL_IDS`: A pair of unique identifiers for the models being evaluated, used to track and distinguish between different model evaluations.
- `MODEL_NAME_OR_PATHS`: A pair of local paths or Hugging Face links to the model weights.
- `CHAT_TEMPLATES`: A pair of chat templates, each corresponding to one of the models.