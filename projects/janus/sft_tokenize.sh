INPUT_PATH=""
OUTPUT_PATH=""
MODEL_PATH=""
CACHE_DIR=""
JANUS_REPO_PATH=""
mkdir -p $CACHE_DIR
NUM_PROCESSES=8
NUM_GPUS=8

export PYTHONPATH=$PYTHONPATH:"$JANUS_REPO_PATH"

python supervised_t2i.py \
    --input_path $INPUT_PATH \
    --output_path $OUTPUT_PATH \
    --model_path $MODEL_PATH \
    --cache_dir $CACHE_DIR \
    --num_processes $NUM_PROCESSES \
    --num_gpus $NUM_GPUS