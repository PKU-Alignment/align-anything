# Support for Rule-based Remote Reward Model

`run_reward_server.py` is used to start the rule-based remote reward server. See `example_start_remote_rm.sh` for further details.

`reward_server.py` is used to interactively start the reward server and initialize localhost:6000, which includes the get_reward function for API endpoint for handling reward requests.

`reward_functions.py` contains custom rule-based reward functions, which can be evaluated based on golden responses (Optional). **You can modify this part for self-registration.**

`remote_rm_client.py` is the remote reward model client, which interacts with the remote reward service via HTTP API. It contains `RemoteRewardModel` class for training.

Input Format [list]: prompts, responses, and golden_responses(optional)
Return [list]: rewards 

TODO: Need to support batch reward evaluation (this can actually be implemented in training reward).

NOTE: 
1. You need to add some extensions:
```
Levenshtein
flask
latex2sympy2_extended
math_verify
```

2. Corner Case Analysis: If the format is incorrect but the answer is correct, the acc rewards will be 0, indicating that correct answers without the proper format are not rewarded.


## An Example

```
export REWARD_PORT=6000
export REWARD_TYPE="example_math"  # Optional: example_math, example_coding, example_safety
export OUTPUT_DIR="./debug_logs"
export DATASET_PATH="./processed_math_data_debug_remote_rm.json"
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
```
