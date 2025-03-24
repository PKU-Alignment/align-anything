export REWARD_PORT=6000
export REWARD_TYPE="example_math"  # Optional: example_math, example_coding, example_safety
export OUTPUT_DIR="./debug_logs"
# Ensure the output directory exists
mkdir -p $OUTPUT_DIR

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
    > $OUTPUT_DIR/reward_server.log 2>&1 &


REWARD_SERVER_PID=$!
echo "Reward server process ID: $REWARD_SERVER_PID"

# 2. Check the log to confirm if the server started successfully
TIMEOUT=20
echo "Waiting $TIMEOUT s for the reward server to start..."
sleep $TIMEOUT  
if grep -q "Running on" $OUTPUT_DIR/reward_server.log; then
    echo "Reward server started successfully."
else
    echo "Failed to start reward server. Check the log for details."
    cat $OUTPUT_DIR/reward_server.log
    kill $REWARD_SERVER_PID
fi
