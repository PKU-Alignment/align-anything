# api token 生成代码
import time
import jwt
import os

minutes = 525600


def encode_jwt_token(ak, sk):
    headers = {
        "alg": "HS256",
        "typ": "JWT"
    }
    payload = {
        "iss": ak,
        "exp": int(time.time()) + minutes,  # 填写您期望的有效时间，此处示例代表当前时间+一年
        "nbf": int(time.time()) - 5  # 填写您期望的生效时间，此处示例代表当前时间-5秒
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
