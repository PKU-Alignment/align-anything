# Support for Rule-based Remote Reward Model

`run_reward_server.py` is used to start the rule-based remote reward server. See `example_start_remote_rm.sh` for further details.

`reward_server.py` is used to interactively start the reward server and initialize localhost:6000, which includes the get_reward function for API endpoint for handling reward requests.

`reward_functions.py` contains custom rule-based reward functions, which can be evaluated based on golden responses (Optional). **You can modify this part for self-registration.**

`remote_rm_client.py` is the remote reward model client, which interacts with the remote reward service via HTTP API. It contains `RemoteRewardModel` class for training.

Input Format [list]: prompts, responses, and golden_responses(optional)
Return [list]: rewards 

TODO: Need to support batch reward evaluation (this can actually be implemented in training reward).
