# Copyright 2024 PKU-Alignment Team and Lagent Team. All Rights Reserved.
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

import os
import time

import jwt


minutes = 525600


def encode_jwt_token(ak, sk):
    headers = {'alg': 'HS256', 'typ': 'JWT'}
    payload = {
        'iss': ak,
        'exp': int(time.time()) + minutes,
        'nbf': int(time.time()) - 5,
    }
    token = jwt.encode(payload, sk, headers=headers)
    return token


def auto_gen_jwt_token(ak, sk):
    token = encode_jwt_token(ak, sk)
    return token


if __name__ == '__main__':
    ak = os.getenv('NOVA_AK')
    sk = os.getenv('NOVA_SK')
    token = encode_jwt_token(ak, sk)
    print(token)
