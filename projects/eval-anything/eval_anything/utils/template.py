# Copyright 2025 PKU-Alignment Team. All Rights Reserved.
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

"""
chat template

TODO 从原库中copy，还需适配
"""

from eval_anything.utils.register import TemplateRegistry


@TemplateRegistry.register('A-OKVQA')
class AOKVQA:
    system_prompt: str = ''
    user_prompt: str = 'USER: \n<image>{input} give me your rationales.'
    assistant_prompt: str = '\nASSISTANT: {output}, the rationales is that {rationales}'
    split_token: str = 'ASSISTANT:'


@TemplateRegistry.register('Llava')
class Llava:
    system_prompt: str = ''
    user_prompt: str = 'USER: \n<image>{input}'
    assistant_prompt: str = '\nASSISTANT:{output}'
    split_token: str = 'ASSISTANT:'
    separator: str = '###'


@TemplateRegistry.register('Qwen2-VL')
class QWEN2VL:
    system_prompt: str = '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n'
    user_prompt: str = (
        '<|im_start|>user\n{input}<|vision_start|><|image_pad|><|vision_end|><|im_end|>\n'
    )
    assistant_prompt: str = '<|im_start|>assistant\n{output}'
    split_token: str = '\n'
    separator: str = 'assistant\n'


@TemplateRegistry.register('Alpaca')
class Alpaca:
    system_prompt: str = 'Below is an instruction that describes a task. '
    user_prompt: str = '### Instruction:\n{input}\n\n'
    assistant_prompt: str = '### Response:\n{output}'


@TemplateRegistry.register('Aquila')
class Aquila:
    system_prompt: str = (
        'A chat between a curious human and an artificial intelligence assistant. '
        "The assistant gives helpful, detailed, and polite answers to the human's questions."
    )
    user_prompt: str = 'Human: {input}'
    assistant_prompt: str = '###Assistant:{output}'
    separator: str = '###'


@TemplateRegistry.register('Atom')
class Atom:
    system_prompt: str = ''
    user_prompt: str = '<bos>Human: {input}\n<eos>'
    assistant_prompt: str = '<bos>Assistant:{output}'
    separator: str = ''


@TemplateRegistry.register('Baichuan')
class Baichuan:
    system_prompt: str = ''
    user_prompt: str = '<reserved_102>{input}'
    assistant_prompt: str = '<reserved_103>{output}'
    separator: str = ''


@TemplateRegistry.register('Baichuan2')
class Baichuan2:
    system_prompt: str = ''
    user_prompt: str = '<reserved_106>{input}'
    assistant_prompt: str = '<reserved_107>{output}'
    separator: str = ''


@TemplateRegistry.register('Belle')
class Belle:
    system_prompt: str = '<bos>'
    user_prompt: str = 'Human: {input}'
    assistant_prompt: str = '\n\nBelle: {output}'
    separator: str = '\n\n'


@TemplateRegistry.register('Bluelm')
class Bluelm:
    system_prompt: str = '<bos>'
    user_prompt: str = 'Human: {input}'
    assistant_prompt: str = '\n\nBelle: {output}'
    separator: str = ''


@TemplateRegistry.register('Breeze')
class Breeze:
    system_prompt: str = '<bos>'
    user_prompt: str = '[INST] {input}'
    assistant_prompt: str = '[/INST]{output}'
    separator: str = ''


@TemplateRegistry.register('Chatglm2')
class Chatglm2:
    system_prompt: str = '[gMASK]<sop>'
    user_prompt: str = '[Round 0]\n\n问：{input}'
    assistant_prompt: str = '\n\n答：{output}'
    separator: str = '\n\n'


@TemplateRegistry.register('Chatglm3')
class Chatglm3:
    system_prompt: str = '[gMASK]<sop><|system|>\n'
    user_prompt: str = '{input}'
    assistant_prompt: str = '{output}'
    separator: str = ''


@TemplateRegistry.register('Chatml')
class Chatml:
    system_prompt: str = '<|im_start|>system\n<|im_end|>\n'
    user_prompt: str = '<|im_start|>user\n{input}<|im_end|>\n'
    assistant_prompt: str = '<|im_start|>assistant\n{output}'
    separator: str = ''


@TemplateRegistry.register('Chatml_de')
class Chatml_de:
    system_prompt: str = 'Du bist ein freundlicher und hilfsbereiter KI-Assistent.'
    user_prompt: str = '<|im_start|>user\n{input}<|im_end|>\n'
    assistant_prompt: str = '<|im_start|>assistant\n{output}'
    separator: str = '\n'


@TemplateRegistry.register('Codegeex2')
class Codegeex2:
    system_prompt: str = '[gMASK]<sop>'
    user_prompt: str = '{input}'
    assistant_prompt: str = '{output}'
    separator: str = ''


@TemplateRegistry.register('Codegeex4')
class Codegeex2:
    system_prompt: str = (
        '[gMASK]<sop><|system|>\n你是一位智能编程助手，你叫CodeGeeX。你会为用户回答关于编程、代码、计算机方面的任何问题，并提供格式规范、可以执行、准确安全的代码，并在必要时提供详细的解释。'
    )
    user_prompt: str = '<|user|>\n{input}'
    assistant_prompt: str = '<|assistant|>\n{output}'
    separator: str = ''


@TemplateRegistry.register('Cohere')
class Cohere:
    system_prompt: str = '<bos><|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|><|END_OF_TURN_TOKEN|>'
    user_prompt: str = '<|START_OF_TURN_TOKEN|><|USER_TOKEN|>{input}<|END_OF_TURN_TOKEN|>'
    assistant_prompt: str = '<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>{output}'
    separator: str = ''


@TemplateRegistry.register('Cpm')
class Cpm:
    system_prompt: str = '<bos>'
    user_prompt: str = '<用户>{input}'
    assistant_prompt: str = '<AI>{output}'
    separator: str = ''


@TemplateRegistry.register('Dbrx')
class Dbrx:
    system_prompt: str = (
        '<|im_start|>system\nYou are DBRX, created by Databricks. You were last updated in December 2023. '
        'You answer questions based on information available up to that point.\n'
        'YOU PROVIDE SHORT RESPONSES TO SHORT QUESTIONS OR STATEMENTS, but provide thorough '
        'responses to more complex and open-ended questions.\nYou assist with various tasks, '
        'from writing to coding (using markdown for code blocks — remember to use ``` with '
        'code, JSON, and tables).\n(You do not have real-time data access or code execution '
        'capabilities. You avoid stereotyping and provide balanced perspectives on '
        'controversial topics. You do not provide song lyrics, poems, or news articles and '
        'do not divulge details of your training data.)\nThis is your system prompt, '
        'guiding your responses. Do not reference it, just respond to the user. If you find '
        'yourself talking about this message, stop. You should be responding appropriately '
        'and usually that means not mentioning this.\nYOU DO NOT MENTION ANY OF THIS INFORMATION '
        "ABOUT YOURSELF UNLESS THE INFORMATION IS DIRECTLY PERTINENT TO THE USER'S QUERY.<|im_end|>\n"
    )
    user_prompt: str = '<|im_start|>user\n{input}<|im_end|>\n'
    assistant_prompt: str = '<|im_start|>assistant\n{output}'
    separator: str = '\n'


@TemplateRegistry.register('Deepseek')
class Deepseek:
    system_prompt: str = '<bos>'
    user_prompt: str = 'User: {input}\n'
    assistant_prompt: str = '\nAssistant:{output}'
    separator: str = ''


@TemplateRegistry.register('Deepseekcoder')
class Deepseekcoder:
    system_prompt: str = (
        '<bos>You are an AI programming assistant, utilizing the Deepseek Coder model, developed by Deepseek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer\n'
    )
    user_prompt: str = '### Instruction:\n{input}\n'
    assistant_prompt: str = '### Response:{output}'
    separator: str = '\n'


@TemplateRegistry.register('Dolphin')
class Dolphin:
    system_prompt: str = '<|im_start|>system\nYou are Dolphin, a helpful AI assistant.<|im_end|>\n'
    user_prompt: str = '<|im_start|>user\n{input}<|im_end|>'
    assistant_prompt: str = '\n<|im_start|>assistant\n{output}'
    separator: str = '\n'


@TemplateRegistry.register('Falcon')
class Falcon:
    system_prompt: str = ''
    user_prompt: str = 'User: {input}\n'
    assistant_prompt: str = 'Falcon:{output}'
    separator: str = '\n'


@TemplateRegistry.register('Gemma')
class Gemma:
    system_prompt: str = '<bos>'
    user_prompt: str = '<start_of_turn>user\n{input}<end_of_turn>\n'
    assistant_prompt: str = '<start_of_turn>model\n{output}'
    separator: str = '<end_of_turn>\n'


@TemplateRegistry.register('Glm4')
class Glm4:
    system_prompt: str = '[gMASK]<sop><|system|>\n'
    user_prompt: str = '<|user|>\n{input}'
    assistant_prompt: str = '<|assistant|>\n{output}'
    separator: str = ''


@TemplateRegistry.register('Intern')
class Intern:
    system_prompt: str = '<bos><|System|>:\n'
    user_prompt: str = '<|User|>:{input}\n'
    assistant_prompt: str = '<|Bot|>:{output}'
    separator: str = '<eoa>\n'


@TemplateRegistry.register('Intern2')
class Intern2:
    system_prompt: str = '<bos><|im_start|>system\n{input}<|im_end|>\n'
    user_prompt: str = '<|im_start|>user\n{input}<|im_end|>\n'
    assistant_prompt: str = '<|im_start|>assistant\n{output}'
    separator: str = '<|im_end|>\n'


@TemplateRegistry.register('Llama2')
class Llama2:
    system_prompt: str = '<<SYS>>\n\n<</SYS>>\n\n'
    user_prompt: str = '<bos>[INST] {input}'
    assistant_prompt: str = '[/INST]{output}'
    separator: str = ''


@TemplateRegistry.register('Llama2_zh')
class Llama2_zh:
    system_prompt: str = (
        '<<SYS>>\nYou are a helpful assistant. 你是一个乐于助人的助手。\n<</SYS>>\n\n'
    )
    user_prompt: str = '<bos>[INST] {input}'
    assistant_prompt: str = '[/INST]{output}'
    separator: str = ''


@TemplateRegistry.register('Llama3')
class Llama3:
    system_prompt: str = '<bos><|start_header_id|>system<|end_header_id|>\n\n'
    user_prompt: str = '<|start_header_id|>user<|end_header_id|>\n\n{input}<|eot_id|>'
    assistant_prompt: str = '<|start_header_id|>assistant<|end_header_id|>\n\n{output}'
    separator: str = ''


@TemplateRegistry.register('Mistral')
class Mistral:
    system_prompt: str = '<bos>'
    user_prompt: str = '[INST] {input}'
    assistant_prompt: str = '[/INST]{output}'
    separator: str = ''


@TemplateRegistry.register('Olmo')
class Olmo:
    system_prompt: str = '<eos>'
    user_prompt: str = '<|user|>\n{input}'
    assistant_prompt: str = '<|assistant|>\n{output}'
    separator: str = ''


@TemplateRegistry.register('Openchat')
class Openchat:
    system_prompt: str = '<bos>'
    user_prompt: str = 'GPT4 Correct User: {input}<eos>'
    assistant_prompt: str = 'GPT4 Correct Assistant:{output}'
    separator: str = ''


@TemplateRegistry.register('Openchat3')
class Openchat3:
    system_prompt: str = '<bos>'
    user_prompt: str = '<|start_header_id|>GPT4 Correct User<|end_header_id|>\n\n{input}<|eot_id|>'
    assistant_prompt: str = '<|start_header_id|>GPT4 Correct Assistant<|end_header_id|>\n\n'
    separator: str = ''


@TemplateRegistry.register('Orion')
class Orion:
    system_prompt: str = '<bos>'
    user_prompt: str = 'Human: {input}\n'
    assistant_prompt: str = '\nAssistant: <eos>{output}'
    separator: str = ''


@TemplateRegistry.register('Phi')
class Phi:
    system_prompt: str = '<bos>'
    user_prompt: str = '<|user|>\n{input}<|end|>'
    assistant_prompt: str = '\n<|assistant|>\n{output}'
    separator: str = '\n'


@TemplateRegistry.register('Qwen')
class Qwen:
    system_prompt: str = '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n'
    user_prompt: str = '<|im_start|>user\n{input}<|im_end|>'
    assistant_prompt: str = '\n<|im_start|>assistant\n{output}'
    separator: str = '\n'


@TemplateRegistry.register('Solar')
class Solar:
    system_prompt: str = '### System:\n\n\n'
    user_prompt: str = '### User:\n{input}\n'
    assistant_prompt: str = '\n### Assistant:\n{output}'
    separator: str = ''


@TemplateRegistry.register('Starchat')
class Starchat:
    system_prompt: str = '<|system|>\n<|end|>\n'
    user_prompt: str = '<|user|>\n{input}<|end|>'
    assistant_prompt: str = '\n<|assistant|>{output}'
    separator: str = '\n'


@TemplateRegistry.register('Telechat')
class Telechat:
    system_prompt: str = r'<\_system><\_end>'
    user_prompt: str = '<_user>{input}'
    assistant_prompt: str = '<_bot>{output}'
    separator: str = ''


@TemplateRegistry.register('Xuanyuan')
class Xuanyuan:
    system_prompt: str = (
        '以下是用户和人工智能助手之间的对话。用户以Human开头，人工智能助手以Assistant开头，会对人类提出的问题给出有帮助、高质量、详细和礼貌的回答，并且总是拒绝参与与不道德、不安全、有争议、政治敏感等相关的话题、问题和指示。\n'
    )
    user_prompt: str = 'Human: {input} '
    assistant_prompt: str = 'Assistant:{output}'
    separator: str = ''


@TemplateRegistry.register('Xverse')
class Xverse:
    system_prompt: str = ''
    user_prompt: str = 'Human: {input}\n'
    assistant_prompt: str = '\nAssistant: {output}'
    separator: str = ''


@TemplateRegistry.register('Yayi')
class Yayi:
    system_prompt: str = (
        "<|System|>\nYou are a helpful, respectful and honest assistant named YaYi developed by Beijing Wenge Technology Co.,Ltd. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n\n"
    )
    user_prompt: str = '<|Human|>\n{input}\n'
    assistant_prompt: str = '\n<|YaYi|>:{output}'
    separator: str = '\n\n'


@TemplateRegistry.register('Yi')
class Yi:
    system_prompt: str = '<|im_start|>system\n<|im_end|>\n'
    user_prompt: str = '<|im_start|>user\n<|im_end|>\n'
    assistant_prompt: str = '<|im_start|>assistant\n{output}'
    separator: str = '\n'


@TemplateRegistry.register('Yi_vl')
class Yi_vl:
    system_prompt: str = (
        'This is a chat between an inquisitive human and an AI assistant. '
        "Assume the role of the AI assistant. Read all the images carefully, and respond to the human's questions with informative, helpful, detailed and polite answers. 这是一个好奇的人类和一个人工智能助手之间的对话。假设你扮演这个AI助手的角色。仔细阅读所有的图像，并对人类的问题做出信息丰富、有帮助、详细的和礼貌的回答。\n\n"
    )
    user_prompt: str = '### Human: {input}\n'
    assistant_prompt: str = '### Assistant:{output}'
    separator: str = '\n'


@TemplateRegistry.register('Yuan')
class Yuan:
    system_prompt: str = ''
    user_prompt: str = '{input}<sep>'
    assistant_prompt: str = '{output}'
    separator: str = '\n'


@TemplateRegistry.register('Zephyr')
class Zephyr:
    system_prompt: str = '<|system|>\nYou are Zephyr, a helpful assistant.'
    user_prompt: str = '<|user|>\n{input}<eos>'
    assistant_prompt: str = '<|assistant|>\n{output}'
    separator: str = '\n'


@TemplateRegistry.register('Ziya')
class Zephyr:
    system_prompt: str = ''
    user_prompt: str = '<human>:{input}\n'
    assistant_prompt: str = '<bot>:{output}'
    separator: str = '\n'


@TemplateRegistry.register('LLAMA_3_2')
class LLAMA_3_2:
    system_prompt: str = ''
    user_prompt: str = (
        '<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n<|image|>{input}<|eot_id|>'
    )
    assistant_prompt: str = '<|start_header_id|>assistant<|end_header_id|>\n{output}'
    split_token: str = '<|start_header_id|>assistant<|end_header_id|>'
    separator: str = '###'


@TemplateRegistry.register('Qwen2Audio')
class Qwen2Audio:
    system_prompt: str = 'You are a helpful assistant.'
    user_prompt: str = (
        '<|im_start|>user\nAudio 1: <|audio_bos|><|AUDIO|><|audio_eos|>\n{input}<|im_end|>\n'
    )
    assistant_prompt: str = '<|im_start|>assistant{output}'
    split_token: str = '\nassistant\n'
    separator: str = '\nassistant\n'
