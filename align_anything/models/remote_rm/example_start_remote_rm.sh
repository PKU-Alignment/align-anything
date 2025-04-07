export REWARD_PORT=6000
export REWARD_TYPE="math_verifier"  # Optional: example_math, example_coding, example_safety, math_verifier(For Zero RL)
export OUTPUT_DIR="./debug_logs"
export DATASET_PATH="./math_verify_dataset/mathvl_345_example.json"
# Ensure the output directory exists
mkdir -p $OUTPUT_DIR

# NOTE The golden dataset should in the keys (question, answer)
# NOTE But when you use the reward server, you should use the keys (prompts, responses)

# 0. Check if the reward port is available
if REWARD_SERVER_PID=$(lsof -t -i:$REWARD_PORT); then
    echo "Remote reward server is already running on port $REWARD_PORT."
    echo "Killing the existing server..."
    kill $REWARD_SERVER_PID
fi

# 1. Start remote reward server (run in the background)
echo "Start remote reward server..."
python -m align_anything.models.remote_rm.run_reward_server \
    --reward-type $REWARD_TYPE \
    --port $REWARD_PORT \
    --dataset $DATASET_PATH \
    > $OUTPUT_DIR/reward_server.log 2>&1 &


REWARD_SERVER_PID=$!
echo "Reward server process ID: $REWARD_SERVER_PID"

# 2. Check the log to confirm if the server started successfully
TIMEOUT=20
echo "Waiting $TIMEOUT s for the reward server to start..."
sleep $TIMEOUT
if grep -q "Running on" $OUTPUT_DIR/reward_server.log; then
    echo "Reward server started successfully."
    cat $OUTPUT_DIR/reward_server.log
else
    echo "Failed to start reward server. Check the log for details."
    cat $OUTPUT_DIR/reward_server.log
    kill $REWARD_SERVER_PID
fi
