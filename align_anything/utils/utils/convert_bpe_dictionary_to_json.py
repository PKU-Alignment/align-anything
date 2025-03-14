# Copyright 2024 Allen Institute for AI

# Copyright 2024-2025 Align-Anything Team. All Rights Reserved.
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


import json


action_dict = {
    'end': 50000,
    'l': 43979,
    'r': 43407,
    'ml': 28026,
    'mr': 24467,
    'mml': 23708,
    'mmr': 22301,
    'mmml': 19873,
    'mmmmmmmr': 18045,
    'mmmmmml': 17989,
    'mmmmmmml': 17262,
    'mmmml': 16681,
    'mmmmml': 15435,
    'mmmmr': 15368,
    'mmmmmmr': 15303,
    'mrr': 15210,
    'mmmr': 15185,
    'mll': 14420,
    'mmmmmr': 13665,
    'rs': 13480,
    'mmmmmmmmr': 12670,
    'mmmmmmmmmmmmmmmm': 12651,
    'll': 12644,
    'rr': 12213,
    'ls': 12118,
    'mmmmmmmml': 12095,
    'b': 10295,
    'mmrr': 9939,
    'mmmmmmmmmr': 9024,
    'mmll': 8800,
    'mmmmmmmmml': 8725,
    'mmmll': 8631,
    'lll': 8518,
    'rrr': 8381,
    'mmmrr': 8374,
    'mmmmmmmmmmr': 7982,
    'mmmmmmmmmml': 7828,
    'mmmmrr': 7579,
    'm': 7463,
    'mmmmll': 7195,
    'llll': 7156,
    'mmmmrmml': 7050,
    'rrrr': 7018,
    'mm': 6974,
    'rrrrr': 6866,
    'mmmmmrr': 6641,
    'lllll': 6613,
    'mmmmmrmml': 6538,
    'mmmrmml': 6515,
    'mrmmr': 6355,
    'mlml': 6321,
    'mrmr': 6255,
    'mmrmmr': 6181,
    'mmlmml': 6115,
    'mmrmml': 6092,
    'mmmlmmr': 6005,
    'mmmmrmmr': 5946,
    'mlmml': 5919,
    'mmrmr': 5899,
    'mmmmmmmmmmml': 5871,
    'mmmmrmr': 5868,
    'mmmmmmmmmmmr': 5867,
    'mmrmmml': 5855,
    'mmlml': 5817,
    'mmmmmlmmr': 5805,
    'mmmmrml': 5725,
    'mmlmmmr': 5581,
    'mmmmmll': 5579,
    'mmmmlmmr': 5523,
    'mmmmlmml': 5517,
    'mmlmmr': 5422,
    'mmmmlml': 5359,
    'mmmmrmmml': 5318,
    'mmmmmmll': 5314,
    'mrrr': 5279,
    'mmmlml': 5182,
    'mlmmr': 5152,
    'mmmlmml': 5109,
    'mrmml': 5100,
    'mmmmmmrr': 5098,
    'mmmrmr': 4975,
    'mlmmmr': 4969,
    'mmmm': 4860,
    'mrmmml': 4856,
    'rmr': 4851,
    'lml': 4836,
    'mmrmmmr': 4817,
    'mrmmmr': 4734,
    'mmmrml': 4727,
    'mlll': 4694,
    'mmmmmrmmml': 4692,
    'mmmrmmr': 4674,
    'mmlmmml': 4655,
    'mmmmlmr': 4557,
    'mmm': 4548,
    'mmmmlmmmr': 4413,
    'mmmmmmmmmmmmr': 4382,
    'mmmlmmmr': 4350,
    'mmmmmmmmmmmml': 4326,
    'mmmrmmml': 4275,
    'mlmmml': 4232,
    'mmmlmr': 4193,
    'mmmmrmll': 4147,
    'mmmmmlmmmr': 4066,
    'mmmmrmmmr': 4052,
    'mmrml': 4016,
    'mmmmmrmmr': 3957,
    'mmmmmmrmmml': 3883,
    'mmmmmmmll': 3860,
    'mmmrmmmr': 3805,
    'mmmlmmml': 3772,
    'mmmmmrmr': 3660,
    'mmmmmmrml': 3651,
    'mmmmmmmrr': 3643,
    'mmlmr': 3615,
    'mmmmmmrmml': 3593,
    'mmmmmmlmmmr': 3585,
    'mrmmmml': 3475,
    'mmmmmlml': 3447,
    'mmmmmlmml': 3394,
    'mmmmlmrr': 3385,
    'mmmmrmmmml': 3359,
    'mmmmlmmml': 3350,
    'mmmmmrml': 3335,
    'mmmmmmlmmr': 3317,
    'mmmmmmmmmmmmmr': 3294,
    'mmmmmmmmmmmmml': 3281,
    'mmmmlmmmmr': 2973,
    'mmrrr': 2855,
    'mmmmmlmr': 2839,
    'mmmmmrmmmr': 2801,
    'mmlll': 2779,
    'mmmrmmmml': 2772,
    'mmmmmmmmll': 2613,
    'mmrmll': 2584,
    'mmmmmmmmrr': 2583,
    'mmmmmmmmmmmmmml': 2572,
    'mmlmrr': 2569,
    'mmmmrmmll': 2568,
    'mmmlmrr': 2538,
    'mmmmmmmmmmmmmmr': 2527,
    'mmmmmrmll': 2438,
    'llml': 2424,
    'mmmrmll': 2408,
    'rrmr': 2400,
    'mmmmlmmrr': 2284,
    'mmmmmlmrr': 2178,
    'mmmmmmmmmll': 2098,
    'mmmmrmmmmml': 2056,
    'mmmmmmmmmrr': 1987,
    'llllll': 1952,
    'mmmmmmmmmmmmmmmr': 1949,
    'mmmmmmmmmmmmmmml': 1923,
    'rrrrrr': 1896,
    'mmmmmmmmmmmmmmmmr': 1532,
    'mmmmmmmmmmmmmmmml': 1490,
    'mls': 1468,
    'mrs': 1462,
    'mmmmlll': 1458,
    'mmmmrrr': 1454,
    'mmmmmlll': 1319,
    'mmmmmmmm': 1282,
    'mmls': 1163,
    'mmllmml': 1107,
    'mmmrmlll': 871,
    'mmmlmrrr': 779,
}

# List of all action atoms including combined ones
action_atoms = ['m', 'r', 'l', 'rs', 'ls', 'end', 'b']


# Function to insert hyphens between individual action atoms
def insert_hyphens(key):
    # Start with the longest matches to avoid misinterpreting substrings
    i = 0
    result = []
    while i < len(key):
        # Check for two-character atoms first ('rs' and 'ls')
        if key[i : i + 2] in action_atoms:
            result.append(key[i : i + 2] + '-')
            i += 2  # Move past the two characters
        elif key[i] in action_atoms:
            result.append(key[i] + '-')
            i += 1  # Move past the single character
        else:
            i += 1  # Move past unrecognized characters
    return ''.join(result).rstrip('-')  # Join all parts and remove trailing dash


# Create a new dictionary with hyphenated keys
new_action_dict = {}
for key, value in action_dict.items():
    new_key = insert_hyphens(key)
    new_action_dict[new_key] = value

action_dict = new_action_dict

# save the dictionary to a json file
with open('utils/action_dict.json', 'w') as f:
    json.dump(action_dict, f, indent=4)
