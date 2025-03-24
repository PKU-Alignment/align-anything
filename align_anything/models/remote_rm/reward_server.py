# Copyright 2025 PKU-Alignment Team. All Rights Reserved.
#
# This code is inspired by the lmm-r1 library.
# https://github.com/TideDra/lmm-r1/tree/main/openrlhf/models/remote_rm/math_verifier.py
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

import argparse
import json
import os
from typing import Any, Dict, List, Optional, Union, Callable

from flask import Flask, jsonify, request

app = Flask(__name__)

# Global variable to store the reward function
reward_function = None

def default_reward_function(prompt: List[str], response: List[str], golden_response: Optional[List[str]] = None) -> List[float]:
    """
    Default reward function implementation, can be replaced by specific task reward functions
    
    Args:
        prompt: List of input prompts
        response: List of model responses
        golden_response: Optional list of golden responses
    Returns:
        List of reward scores for each (prompt, response) pair
    """
    # Simple example: response length as reward (only for demonstration)
    return [min(len(resp) / 100, 1.0) for resp in response]

@app.route("/get_reward", methods=["POST"])
def get_reward():
    """API endpoint for handling reward requests"""
    try:
        data = request.get_json()
        
        # Check necessary fields
        if "prompts" not in data or "responses" not in data:
            return jsonify({"error": "Request must contain 'prompts' and 'responses' fields, optional 'golden_responses' field"}), 400
            
        prompts = data["prompts"]
        responses = data["responses"]
        golden_responses = data.get("golden_responses", None)
        # Check data validity
        if len(prompts) != len(responses):
            return jsonify({"error": "The number of prompts and responses must be the same"}), 400
        print(f"Received prompts: {prompts}")
        print(f"Received responses: {responses}")
        print(f"Received golden_responses: {golden_responses}")
        # Calculate rewards
        rewards = reward_function(prompts, responses, golden_responses)
        
        # Return reward results
        return jsonify({"rewards": rewards})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# NOTE Feature: You can register your own reward function here
def register_reward_function(func: Callable[[List[str], List[str], Optional[List[str]]], List[float]]):
    """Register custom reward function"""
    global reward_function
    reward_function = func
    print(f"Registered reward function: {func.__name__}")

def start_server(host: str = "0.0.0.0", port: int = 6000, reward_func: Optional[Callable] = None):
    """
    Start reward server
    
    Args:
        host: Server host address
        port: Server port
        reward_func: Custom reward function
    """
    global reward_function
    # print(f"Reward function: {reward_func}")
    # Register reward function
    if reward_func is not None:
        register_reward_function(reward_func)
    else:
        register_reward_function(default_reward_function)
        
    # Start server
    print(f"Reward server started at http://{host}:{port}")
    app.run(host=host, port=port, debug=False, use_reloader=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start rule-based reward model server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host address")
    parser.add_argument("--port", type=int, default=6000, help="Server port")
    parser.add_argument("--reward_func", type=str, help="Reward function")
    
    args = parser.parse_args()
    start_server(host=args.host, port=args.port, reward_func=args.reward_func)