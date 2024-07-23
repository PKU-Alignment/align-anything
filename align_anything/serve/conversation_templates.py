
from typing import List, Any, Dict, Union, Tuple
from enum import auto, IntEnum
import dataclasses
import math
import os
import re
import sys

ANTHROPIC_MODEL_LIST = (
    "claude-1",
    "claude-2",
    "claude-2.0",
    "claude-2.1",
    "claude-3-haiku-20240307",
    "claude-3-haiku-20240307-vertex",
    "claude-3-sonnet-20240229",
    "claude-3-sonnet-20240229-vertex",
    "claude-3-5-sonnet-20240620",
    "claude-3-opus-20240229",
    "claude-instant-1",
    "claude-instant-1.2",
)

OPENAI_MODEL_LIST = (
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-0301",
    "gpt-3.5-turbo-0613",
    "gpt-3.5-turbo-1106",
    "gpt-3.5-turbo-0125",
    "gpt-4",
    "gpt-4-0314",
    "gpt-4-0613",
    "gpt-4-turbo",
    "gpt-4-1106-preview",
    "gpt-4-0125-preview",
    "gpt-4-turbo-browsing",
    "gpt-4-turbo-2024-04-09",
    "gpt2-chatbot",
    "im-also-a-good-gpt2-chatbot",
    "im-a-good-gpt2-chatbot",
    "gpt-4o-2024-05-13",
)
class SeparatorStyle(IntEnum):
    """Separator styles."""

    ADD_COLON_SINGLE = auto()
    ADD_COLON_TWO = auto()
    ADD_COLON_SPACE_SINGLE = auto()
    NO_COLON_SINGLE = auto()
    NO_COLON_TWO = auto()
    ADD_NEW_LINE_SINGLE = auto()
    LLAMA2 = auto()
    LLAMA3 = auto()
    CHATGLM = auto()
    CHATML = auto()
    CHATINTERN = auto()
    DOLLY = auto()
    RWKV = auto()
    PHOENIX = auto()
    ROBIN = auto()
    FALCON_CHAT = auto()
    CHATGLM3 = auto()
    DEEPSEEK_CHAT = auto()
    METAMATH = auto()
    YUAN2 = auto()
    GEMMA = auto()
    CLLM = auto()
    DEFAULT = auto()
@dataclasses.dataclass
class Conversation:
    """A class that manages prompt templates and keeps all conversation history."""

    # The name of this template
    name: str
    # The template of the system prompt
    system_template: str = "{system_message}"
    # The system message
    system_message: str = ""
    system_message_vision: str = ""
    # The names of two roles
    roles: Tuple[str] = ("USER", "ASSISTANT")
    # All messages. Each item is (role, message).
    # Each message is either a string or a tuple of (string, List[image_url]).
    messages: List[List[str]] = ()
    # The number of few shot examples
    offset: int = 0
    # The separator style and configurations
    sep_style: SeparatorStyle = SeparatorStyle.ADD_COLON_SINGLE
    sep: str = "\n"
    sep2: str = None
    # Stop criteria (the default one is EOS token)
    stop_str: Union[str, List[str]] = None
    # Stops generation if meeting any token in this list
    stop_token_ids: List[int] = None
    # The maximum image size in megabytes that this model takes in. None means we do not resize the image.
    max_image_size_mb: int = None



    def get_prompt(self, messages) -> str:
        """Get the prompt for generation."""
        self.messages = messages
        system_prompt = self.system_template.format(system_message=self.system_message)
        if self.sep_style == SeparatorStyle.ADD_COLON_SINGLE:
            ret = system_prompt + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + ": " + message + self.sep
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.ADD_COLON_TWO:
            seps = [self.sep, self.sep2]
            ret = system_prompt + seps[0]
            for i, (role, message) in enumerate(self.messages):
                if message:
                    if type(message) is tuple:
                        message, images = message
                        message = IMAGE_PLACEHOLDER_STR * len(images) + message
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.ADD_COLON_SPACE_SINGLE:
            ret = system_prompt + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + ": " + message + self.sep
                else:
                    ret += role + ": "  # must be end with a space
            return ret
        elif self.sep_style == SeparatorStyle.ADD_NEW_LINE_SINGLE:
            ret = "" if system_prompt == "" else system_prompt + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + "\n" + message + self.sep
                else:
                    ret += role + "\n"
            return ret
        elif self.sep_style == SeparatorStyle.NO_COLON_SINGLE:
            ret = system_prompt
            for role, message in self.messages:
                if message:
                    ret += role + message + self.sep
                else:
                    ret += role
            return ret
        elif self.sep_style == SeparatorStyle.NO_COLON_TWO:
            seps = [self.sep, self.sep2]
            ret = system_prompt
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + message + seps[i % 2]
                else:
                    ret += role
            return ret
        elif self.sep_style == SeparatorStyle.RWKV:
            ret = system_prompt
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += (
                        role
                        + ": "
                        + message.replace("\r\n", "\n").replace("\n\n", "\n")
                    )
                    ret += "\n\n"
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.LLAMA2:
            seps = [self.sep, self.sep2]
            if self.system_message:
                ret = system_prompt
            else:
                ret = "[INST] "
            for i, (role, message) in enumerate(self.messages):
                tag = self.roles[i % 2]
                if message:
                    if i == 0:
                        ret += message + " "
                    else:
                        ret += tag + " " + message + seps[i % 2]
                else:
                    ret += tag
            return ret
        elif self.sep_style == SeparatorStyle.LLAMA3:
            ret = "<|begin_of_text|>"
            if self.system_message:
                ret += system_prompt
            else:
                ret += ""
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += f"<|start_header_id|>{role}<|end_header_id|>\n\n"
                    ret += f"{message.strip()}<|eot_id|>"
                else:
                    ret += f"<|start_header_id|>{role}<|end_header_id|>\n\n"
            return ret
        elif self.sep_style == SeparatorStyle.CHATGLM:
            # source: https://huggingface.co/THUDM/chatglm-6b/blob/1d240ba371910e9282298d4592532d7f0f3e9f3e/modeling_chatglm.py#L1302-L1308
            # source2: https://huggingface.co/THUDM/chatglm2-6b/blob/e186c891cf64310ac66ef10a87e6635fa6c2a579/modeling_chatglm.py#L926
            round_add_n = 1 if self.name == "chatglm2" else 0
            if system_prompt:
                ret = system_prompt + self.sep
            else:
                ret = ""

            for i, (role, message) in enumerate(self.messages):
                if i % 2 == 0:
                    ret += f"[Round {i//2 + round_add_n}]{self.sep}"

                if message:
                    ret += f"{role}：{message}{self.sep}"
                else:
                    ret += f"{role}："
            return ret
        elif self.sep_style == SeparatorStyle.CHATML:
            ret = "" if system_prompt == "" else system_prompt + self.sep + "\n"
            for role, message in self.messages:
                if message:
                    if type(message) is tuple:
                        message, images = message
                        message = IMAGE_PLACEHOLDER_STR * len(images) + message
                    ret += role + "\n" + message + self.sep + "\n"
                else:
                    ret += role + "\n"
            return ret
        elif self.sep_style == SeparatorStyle.CHATGLM3:
            ret = ""
            if self.system_message:
                ret += system_prompt
            for role, message in self.messages:
                if message:
                    ret += role + "\n" + message
                else:
                    ret += role
            return ret
        elif self.sep_style == SeparatorStyle.CHATINTERN:
            # source: https://huggingface.co/internlm/internlm-chat-7b-8k/blob/bd546fa984b4b0b86958f56bf37f94aa75ab8831/modeling_internlm.py#L771
            seps = [self.sep, self.sep2]
            ret = system_prompt
            for i, (role, message) in enumerate(self.messages):
                if i % 2 == 0:
                    ret += "<s>"
                if message:
                    ret += role + ":" + message + seps[i % 2] + "\n"
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.DOLLY:
            seps = [self.sep, self.sep2]
            ret = system_prompt
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + ":\n" + message + seps[i % 2]
                    if i % 2 == 1:
                        ret += "\n\n"
                else:
                    ret += role + ":\n"
            return ret
        elif self.sep_style == SeparatorStyle.PHOENIX:
            ret = system_prompt
            for role, message in self.messages:
                if message:
                    ret += role + ": " + "<s>" + message + "</s>"
                else:
                    ret += role + ": " + "<s>"
            return ret
        elif self.sep_style == SeparatorStyle.ROBIN:
            ret = system_prompt + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + ":\n" + message + self.sep
                else:
                    ret += role + ":\n"
            return ret
        elif self.sep_style == SeparatorStyle.FALCON_CHAT:
            ret = ""
            if self.system_message:
                ret += system_prompt + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + ": " + message + self.sep
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.METAMATH:
            ret = "" if system_prompt == "" else system_prompt + self.sep
            for i, (role, message) in enumerate(self.messages):
                # For MetaMath, sep2 is used to prefix the message.
                starting_sep = ":\n" if i % 2 == 0 else ": " + self.sep2
                ending_sep = self.sep if i % 2 == 0 else ""
                if message:
                    ret += role + starting_sep + message + ending_sep
                else:
                    ret += role + starting_sep
            return ret
        elif self.sep_style == SeparatorStyle.DEEPSEEK_CHAT:
            seps = [self.sep, self.sep2]
            ret = system_prompt
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.YUAN2:
            seps = [self.sep, self.sep2]
            ret = ""
            if self.system_message:
                ret += system_prompt + seps[1]
            for _, message in self.messages:
                if message:
                    ret += message + "<n>"
                else:
                    ret += ""
            ret = ret.rstrip("<n>") + seps[0]
            return ret
        elif self.sep_style == SeparatorStyle.GEMMA:
            ret = "<bos>"
            for role, message in self.messages:
                if message:
                    ret += "<start_of_turn>" + role + "\n" + message + self.sep
                else:
                    ret += "<start_of_turn>" + role + "\n"
            return ret
        elif self.sep_style == SeparatorStyle.CLLM:
            seps = [self.sep, self.sep2]
            ret = system_prompt + seps[0]
            for i, (role, message) in enumerate(self.messages[-2:]):
                if message:
                    if type(message) is tuple:
                        message, images = message
                        message = IMAGE_PLACEHOLDER_STR * len(images) + message
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.DEFAULT:
            ret = system_prompt + "\n"
            for role, message in self.messages:
                if message:
                    if type(message) is tuple:
                        message, images = message
                    ret += role + ": " + message + "\n"
                else:
                    ret += role + ":"
            return ret
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

conv_templates: Dict[str, Conversation] = {}


def register_conv_template(template: Conversation, override: bool = False):
    """Register a new conversation template."""
    if not override:
        assert (
            template.name not in conv_templates
        ), f"{template.name} has been registered."

    conv_templates[template.name] = template

def get_conv_template(name: str) -> Conversation:
    """Get the conversation template by name."""
    return conv_templates[name]
def get_template(model_path: str) -> Conversation:
    """Get the conversation template by name."""
    if "vicuna" in model_path.lower():
        if "v0" in remove_parent_directory_name(model_path):
            return get_conv_template("one_shot")
        return get_conv_template("vicuna_v1.1")
    elif  re.search(r"airoboros|spicyboros", model_path, re.I):
        if "-3." in model_path or "-3p" in model_path:
            return get_conv_template("airoboros_v3")
        if "spicyboros" in model_path or re.search(r"-(2\.[2-9]+)", model_path):
            return get_conv_template("airoboros_v2")
        return get_conv_template("airoboros_v1")
    elif "longchat" in model_path.lower():
        return get_conv_template("vicuna_v1.1")
    elif "koala" in model_path.lower():
        return get_conv_template("koala_v1")
    elif "alpaca" in model_path.lower():
        return get_conv_template("alpaca")
    elif "chatglm" in model_path.lower():
        if "chatglm2" in model_path.lower():
            return get_conv_template("chatglm2")
        if "chatglm3" in model_path.lower():
            return get_conv_template("chatglm3")
        return get_conv_template("chatglm")
    elif "codegeex" in model_path.lower():
        return get_conv_template("codegeex")
    elif "dolly-v2" in model_path.lower():
        return get_conv_template("dolly_v2")
    elif "oasst" in model_path and "pythia" in model_path.lower():
        return get_conv_template("oasst_pythia")
    elif ( "openassistant-sft-7-llama-30b-hf" in model_path.lower()) or ("oasst" in model_path.lower() and "pythia" not in model_path.lower()):
        return get_conv_template("oasst_llama")
    elif ("openchat" in model_path.lower() and "3.5" in model_path.lower()) or "starling-lm" in model_path.lower():
        return get_conv_template("openchat_3.5")
    elif "tenyxchat" in model_path.lower():
        return get_conv_template("tenyxchat")
    elif "pythia" in model_path.lower():
        return get_conv_template("stablelm")
    elif "mpt" in model_path.lower() and not "airoboros" in model_path.lower():
        model_path = model_path.lower()
        if "mpt-7b-chat" in model_path:
            return get_conv_template("mpt-7b-chat")
        elif "mpt-30b-chat" in model_path:
            return get_conv_template("mpt-30b-chat")
        elif "mpt-30b-instruct" in model_path:
            return get_conv_template("mpt-30b-instruct")
        else:
            print(
                "Warning: Loading base MPT model with `zero_shot` conversation configuration.  "
                "If this is not desired, inspect model configurations and names."
            )
            return get_conv_template("zero_shot")
    elif "baize" in model_path.lower():
        return get_conv_template("baize")
    elif "rwkv" in model_path.lower():
        return get_conv_template("rwkv")
    elif "openbuddy" in model_path.lower():
        return get_conv_template("openbuddy")
    elif "phoenix" in model_path.lower():
        return get_conv_template("phoenix")
    elif "realm" in model_path.lower():
        return get_conv_template("ReaLM-7b-v1")
    elif model_path in OPENAI_MODEL_LIST:
        if "browsing" in model_path:
            return get_conv_template("api_based_default")
        if "gpt-4-turbo-2024-04-09" in model_path:
            return get_conv_template("gpt-4-turbo-2024-04-09")
        if "gpt2-chatbot" in model_path:
            return get_conv_template("gpt-4-turbo-2024-04-09")
        if "gpt-4o" in model_path:
            return get_conv_template("gpt-4-turbo-2024-04-09")
        return get_conv_template("chatgpt")
    elif model_path.lower() in ("azure-gpt-35-turbo", "azure-gpt-4"):
        return get_conv_template("chatgpt")
    elif model_path.lower() in (
            "pplx-7b-online",
            "pplx-70b-online",
        ):
        return get_conv_template("pplxai")
    elif model_path in ANTHROPIC_MODEL_LIST:
        if "claude-3-haiku" in model_path:
            return get_conv_template("claude-3-haiku-20240307")
        if "claude-3-sonnet" in model_path:
            return get_conv_template("claude-3-sonnet-20240229")
        if "claude-3-5-sonnet" in model_path:
            return get_conv_template("claude-3-5-sonnet-20240620")
        if "claude-3-opus" in model_path:
            return get_conv_template("claude-3-opus-20240229")
        return get_conv_template("claude")
    elif "bard" in model_path.lower():
        return get_conv_template("bard")
    elif "palm-2" in model_path.lower():
        return get_conv_template("bard")
    elif "gemini" in model_path.lower():
        return get_conv_template("gemini")
    elif "gemini-1.5-pro" in model_path.lower():
        return get_conv_template("gemini-dev")
    elif "billa" in model_path.lower():
        return get_conv_template("billa")
    elif "redpajama-incite" in model_path.lower():
        return get_conv_template("redpajama-incite")
    elif "h2ogpt" in model_path.lower():
        return get_conv_template("h2ogpt")
    elif "robin" in model_path.lower():
        return get_conv_template("Robin")
    elif "gpt4all" in model_path.lower() and "snoozy" in model_path.lower():
        return get_conv_template("snoozy")
    elif "wizardlm" in model_path.lower():
        if "13b" in model_path.lower() or "30b" in model_path.lower() or "70b" in model_path.lower():
            return get_conv_template("vicuna_v1.1")
        else:
            # TODO: use the recommended template for 7B
            # (https://huggingface.co/WizardLM/WizardLM-13B-V1.0)
            return get_conv_template("one_shot")
    elif "manticore" in model_path.lower():
        return get_conv_template("manticore")
    elif  "guanaco" in model_path.lower():
        return get_conv_template("zero_shot")
    elif "polyglot" in model_path.lower() and "chang" in model_path.lower():
        return get_conv_template("polyglot_changgpt")
    elif  "camel" in model_path.lower():
        return get_conv_template("vicuna_v1.1")
    elif "tulu" in model_path.lower():
        return get_conv_template("tulu")
    elif "falcon" in model_path.lower() and "chat" not in model_path.lower():
        return get_conv_template("falcon")
    elif "falcon" in model_path.lower() and "chat" in model_path.lower():
        return get_conv_template("falcon_chat")
    elif "tigerbot" in model_path.lower():
        return get_conv_template("tigerbot")
    elif "baichuan" in model_path.lower():
        if "chat" in model_path.lower():
            if "baichuan2" in model_path.lower():
                return get_conv_template("baichuan2-chat")
            return get_conv_template("baichuan-chat")
        return get_conv_template("zero_shot")
    elif "xgen" in model_path.lower():
        return get_conv_template("xgen")
    elif "nous-hermes" in model_path.lower():
        return get_conv_template("alpaca")
    elif "internlm" in model_path.lower():
        return get_conv_template("internlm-chat")
    elif "starchat" in model_path.lower():
        return get_conv_template("starchat")
    elif  "mistral" in model_path.lower() or "mixtral" in model_path.lower():
        return get_conv_template("mistral")
    elif "llama-2" in model_path.lower():
        return get_conv_template("llama-2")
    elif "llama-3" or "llama3" in model_path.lower():
        return get_conv_template("llama-3")
    elif "cutegpt" in model_path.lower():
        return get_conv_template("cutegpt")
    elif "mistral-7b-openorca" in model_path.lower() or "openorca" in model_path.lower():
        if "mistral-7b-openorca" in model_path.lower():
            return get_conv_template("mistral-7b-openorca")
        return get_conv_template("open-orca")
    elif "dolphin" in model_path.lower() and "mistral" in model_path.lower():
        return get_conv_template("dolphin-2.2.1-mistral-7b")
    elif model_path.lower() in ["openhermes-2.5-mistral-7b", "openhermes-2-mistral-7b"]:
        return get_conv_template("OpenHermes-2.5-Mistral-7B")
    elif any(
            model_str in model_path.lower()
            for model_str in [
                "nous-hermes-2-mixtral-8x7b-dpo",
                "nous-hermes-2-mixtral-8x7b-sft",
            ]
        ):
        return get_conv_template("Nous-Hermes-2-Mixtral-8x7B-DPO")
    elif "wizardcoder" in model_path.lower():
        return get_conv_template("alpaca")
    elif "qwen" in model_path.lower():
        return get_conv_template("qwen-7b-chat")
    elif "smaug" in model_path.lower():
        return get_conv_template("qwen-7b-chat")
    elif "bge" in model_path.lower():
        return get_conv_template("one_shot")
    elif "e5-" in model_path.lower():
        return get_conv_template("one_shot")
    elif "aquila" in model_path.lower():
        if "aquilachat2" in model_path.lower():
            if "16k" in model_path.lower():
                return get_conv_template("aquila")
            elif "34b" in model_path.lower():
                return get_conv_template("aquila-legacy")
            else:
                return get_conv_template("aquila-v1")
        else:
            return get_conv_template("aquila-chat")
    elif "llama2-chinese" in model_path.lower():
        return get_conv_template("llama2-chinese")
    elif  "chinese-alpaca" in model_path.lower():
        return get_conv_template("chinese-alpaca2")
    elif bool(re.search(r"vigogne|vigostral", model_path, re.I)):
        if "chat" in model_path.lower():
            if "vigostral" in model_path.lower():
                return get_conv_template("vigogne_chat_v3")
            return get_conv_template("vigogne_chat_v2")
        return get_conv_template("vigogne_instruct")
    elif "open-llama" in model_path.lower() and "open-instruct" in model_path.lower():
        return get_conv_template("alpaca")
    elif "codellama" in model_path.lower():
        return get_conv_template("llama-2")
    elif "stable-vicuna" in model_path.lower():
        return get_conv_template("stable-vicuna")
    elif "phind-codellama-" in model_path.lower():
        return get_conv_template("phind")
    elif "llama2-ko-chang" in model_path.lower():
        return get_conv_template("polyglot_changgpt")
    elif "zephyr" in model_path.lower():
        return get_conv_template("zephyr")
    elif "notus" in model_path.lower():
        return get_conv_template("zephyr")
    elif "catppt" in model_path.lower():
        return get_conv_template("catppt")
    elif "tinyllama" in model_path.lower():
        return get_conv_template("TinyLlama")
    elif "xwin-lm" in model_path.lower():
        return get_conv_template("vicuna_v1.1")
    elif "lemur-70b-chat" in model_path.lower():
        return get_conv_template("lemur-70b-chat")
    elif bool(
            re.search(r"pygmalion|mythalion|metharme", model_path.lower(), re.I)
        ):
        return get_conv_template("metharme")
    elif "xdan" in model_path.lower() and "v1" in model_path.lower():
        return get_conv_template("xdan-v1")
    elif "orca-2" in model_path.lower():
        return get_conv_template("orca-2")
    elif "yi-" in model_path.lower() and "chat" in model_path.lower():
        return get_conv_template("Yi-34b-chat")
    elif "deepseek-coder" in model_path.lower():
        return get_conv_template("deepseek-coder")
    elif "deepseek-llm" in model_path.lower() and "chat" in model_path.lower():
        return get_conv_template("deepseek-chat")
    elif "gemini" in model_path.lower() or "bard" in model_path.lower():
        if "gemini-1.5-pro" in model_path:
            return get_conv_template("gemini-1.5-pro")
        return get_conv_template("gemini")
    elif "yuan2" in model_path.lower():
        return get_conv_template("yuan2")
    elif "metamath" in model_path.lower():
        return get_conv_template("metamath")
    elif "bagel" in model_path.lower():
        return get_conv_template("airoboros_v3")
    elif "solar-" in model_path.lower() and "instruct" in model_path.lower():
        return get_conv_template("solar")
    elif "steerlm-chat" in model_path.lower():
        return get_conv_template("steerlm")
    elif "gemma" in model_path.lower():
        return get_conv_template("gemma")
    elif "llava" in model_path.lower():
        model_path = model_path.lower()
        if "34b" in model_path:
            return get_conv_template("llava-chatml")

        return get_conv_template("vicuna_v1.1")
    elif "yuan" in model_path.lower():
        return get_conv_template("yuan")
    elif "olmo" in model_path.lower():
        return get_conv_template("api_based_default")
    elif "yandexgpt" in model_path.lower():
        return get_conv_template("yandexgpt")
    elif "consistency-llm" in model_path.lower():
        return get_conv_template("cllm")
    elif model_path in ["command-r"]:
        return get_conv_template("api_based_default")
    elif model_path in ["dbrx-instruct"]:
        return get_conv_template("api_based_default")
    elif "reka" in model_path.lower():
        return get_conv_template("api_based_default")
    


                

    



    
    



register_conv_template(
    Conversation(
        name="raw",
        system_message="",
        roles=("", ""),
        sep_style=SeparatorStyle.NO_COLON_SINGLE,
        sep="",
    )
)
# A template with a one-shot conversation example
register_conv_template(
    Conversation(
        name="one_shot",
        system_message="A chat between a curious human and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the human's questions.",
        roles=("Human", "Assistant"),
        messages=(
            (
                "Human",
                "Got any creative ideas for a 10 year old’s birthday?",
            ),
            (
                "Assistant",
                """Of course! Here are some creative ideas for a 10-year-old's birthday party:
1. Treasure Hunt: Organize a treasure hunt in your backyard or nearby park. Create clues and riddles for the kids to solve, leading them to hidden treasures and surprises.
2. Science Party: Plan a science-themed party where kids can engage in fun and interactive experiments. You can set up different stations with activities like making slime, erupting volcanoes, or creating simple chemical reactions.
3. Outdoor Movie Night: Set up a backyard movie night with a projector and a large screen or white sheet. Create a cozy seating area with blankets and pillows, and serve popcorn and snacks while the kids enjoy a favorite movie under the stars.
4. DIY Crafts Party: Arrange a craft party where kids can unleash their creativity. Provide a variety of craft supplies like beads, paints, and fabrics, and let them create their own unique masterpieces to take home as party favors.
5. Sports Olympics: Host a mini Olympics event with various sports and games. Set up different stations for activities like sack races, relay races, basketball shooting, and obstacle courses. Give out medals or certificates to the participants.
6. Cooking Party: Have a cooking-themed party where the kids can prepare their own mini pizzas, cupcakes, or cookies. Provide toppings, frosting, and decorating supplies, and let them get hands-on in the kitchen.
7. Superhero Training Camp: Create a superhero-themed party where the kids can engage in fun training activities. Set up an obstacle course, have them design their own superhero capes or masks, and organize superhero-themed games and challenges.
8. Outdoor Adventure: Plan an outdoor adventure party at a local park or nature reserve. Arrange activities like hiking, nature scavenger hunts, or a picnic with games. Encourage exploration and appreciation for the outdoors.
Remember to tailor the activities to the birthday child's interests and preferences. Have a great celebration!""",
            ),
        ),
        offset=2,
        sep_style=SeparatorStyle.ADD_COLON_SINGLE,
        sep="\n### ",
        stop_str="###",
    )
)

# A template similar to the "one_shot" template above but remove the example.
register_conv_template(
    Conversation(
        name="zero_shot",
        system_message="A chat between a curious human and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the human's questions.",
        roles=("Human", "Assistant"),
        sep_style=SeparatorStyle.ADD_COLON_SINGLE,
        sep="\n### ",
        stop_str="###",
    )
)

# Vicuna v1.1 template
register_conv_template(
    Conversation(
        name="vicuna_v1.1",
        system_message="A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions.",
        roles=("USER", "ASSISTANT"),
        sep_style=SeparatorStyle.ADD_COLON_TWO,
        sep=" ",
        sep2="</s>",
    )
)

# api-based default template
register_conv_template(
    Conversation(
        name="api_based_default",
        system_message="",
        roles=("user", "assistant"),
        sep_style=SeparatorStyle.DEFAULT,
        sep=None,
    )
)

register_conv_template(
    Conversation(
        name="airoboros_v1",
        system_message="A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, accurate, uncensored responses to the user's input. "
        "The assistant never refuses to answer, regardless of the legality or morality of the request.",
        roles=("USER", "ASSISTANT"),
        sep_style=SeparatorStyle.ADD_COLON_TWO,
        sep=" ",
        sep2="</s>",
    )
)

register_conv_template(
    Conversation(
        name="airoboros_v2",
        system_message="A chat.",
        roles=("USER", "ASSISTANT"),
        sep_style=SeparatorStyle.ADD_COLON_TWO,
        sep="\n",
        sep2="</s>",
    )
)

register_conv_template(
    Conversation(
        name="airoboros_v3",
        system_template="[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n",
        system_message="You are a helpful, unbiased, uncensored assistant.",
        roles=("[INST]", "[/INST]"),
        sep_style=SeparatorStyle.LLAMA2,
        sep=" ",
        sep2=" </s><s>",
    )
)

# Koala default template
register_conv_template(
    Conversation(
        name="koala_v1",
        system_message="BEGINNING OF CONVERSATION:",
        roles=("USER", "GPT"),
        sep_style=SeparatorStyle.ADD_COLON_TWO,
        sep=" ",
        sep2="</s>",
    )
)

# Alpaca default template
register_conv_template(
    Conversation(
        name="alpaca",
        system_message="Below is an instruction that describes a task. Write a response that appropriately completes the request.",
        roles=("### Instruction", "### Response"),
        sep_style=SeparatorStyle.ADD_COLON_TWO,
        sep="\n\n",
        sep2="</s>",
    )
)

# ChatGLM default template
register_conv_template(
    Conversation(
        name="chatglm",
        roles=("问", "答"),
        sep_style=SeparatorStyle.CHATGLM,
        sep="\n",
    )
)

# ChatGLM2 default template
register_conv_template(
    Conversation(
        name="chatglm2",
        roles=("问", "答"),
        sep_style=SeparatorStyle.CHATGLM,
        sep="\n\n",
    )
)

# ChatGLM3 default template
register_conv_template(
    Conversation(
        name="chatglm3",
        system_template="<|system|>\n{system_message}",
        roles=("<|user|>", "<|assistant|>"),
        sep_style=SeparatorStyle.CHATGLM3,
        stop_token_ids=[
            64795,
            64797,
            2,
        ],  # "<|user|>", "<|observation|>", "</s>"
    )
)

# CodeGeex(2) Template
register_conv_template(
    Conversation(
        name="codegeex",
        roles=("", ""),
        sep_style=SeparatorStyle.NO_COLON_SINGLE,
        sep="\n\n",
        stop_token_ids=[0, 2],
    )
)

# Dolly V2 default template
register_conv_template(
    Conversation(
        name="dolly_v2",
        system_message="Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n",
        roles=("### Instruction", "### Response"),
        sep_style=SeparatorStyle.DOLLY,
        sep="\n\n",
        sep2="### End",
    )
)

# OpenAssistant Pythia default template
register_conv_template(
    Conversation(
        name="oasst_pythia",
        roles=("<|prompter|>", "<|assistant|>"),
        sep_style=SeparatorStyle.NO_COLON_SINGLE,
        sep="<|endoftext|>",
    )
)

# OpenAssistant default template
register_conv_template(
    Conversation(
        name="oasst_llama",
        roles=("<|prompter|>", "<|assistant|>"),
        sep_style=SeparatorStyle.NO_COLON_SINGLE,
        sep="</s>",
    )
)

# OpenChat 3.5 default template
register_conv_template(
    Conversation(
        name="openchat_3.5",
        roles=("GPT4 Correct User", "GPT4 Correct Assistant"),
        sep_style=SeparatorStyle.FALCON_CHAT,
        sep="<|end_of_turn|>",
    )
)

# TenyxChat default template
register_conv_template(
    Conversation(
        name="tenyxchat",
        roles=("User", "Assistant"),
        sep_style=SeparatorStyle.FALCON_CHAT,
        sep="<|end_of_turn|>",
    )
)

# Deepseek code default template
register_conv_template(
    Conversation(
        name="deepseek-coder",
        system_template="You are an AI programming assistant, utilizing the DeepSeek Coder model, developed by DeepSeek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer.",
        roles=("### Instruction:", "### Response:"),
        sep="\n",
        stop_str="<|EOT|>",
        sep_style=SeparatorStyle.ADD_NEW_LINE_SINGLE,
    )
)


# Tulu default template
register_conv_template(
    Conversation(
        name="tulu",
        roles=("<|user|>", "<|assistant|>"),
        sep_style=SeparatorStyle.ADD_NEW_LINE_SINGLE,
        sep="\n",
    )
)

# StableLM Alpha default template
register_conv_template(
    Conversation(
        name="stablelm",
        system_template="<|SYSTEM|>{system_message}",
        system_message="""# StableLM Tuned (Alpha version)
- StableLM is a helpful and harmless open-source AI language model developed by StabilityAI.
- StableLM is excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
- StableLM is more than just an information source, StableLM is also able to write poetry, short stories, and make jokes.
- StableLM will refuse to participate in anything that could harm a human.
""",
        roles=("<|USER|>", "<|ASSISTANT|>"),
        sep_style=SeparatorStyle.NO_COLON_SINGLE,
        sep="",
        stop_token_ids=[50278, 50279, 50277, 1, 0],
    )
)

# Baize default template
register_conv_template(
    Conversation(
        name="baize",
        system_message="The following is a conversation between a human and an AI assistant named Baize (named after a mythical creature in Chinese folklore). Baize is an open-source AI assistant developed by UCSD and Sun Yat-Sen University. The human and the AI assistant take turns chatting. Human statements start with [|Human|] and AI assistant statements start with [|AI|]. The AI assistant always provides responses in as much detail as possible, and in Markdown format. The AI assistant always declines to engage with topics, questions and instructions related to unethical, controversial, or sensitive issues. Complete the transcript in exactly that format.\n",
        roles=("[|Human|]", "[|AI|]"),
        messages=(
            ("[|Human|]", "Hello!"),
            ("[|AI|]", "Hi!"),
        ),
        offset=2,
        sep_style=SeparatorStyle.NO_COLON_SINGLE,
        sep="\n",
        stop_str="[|Human|]",
    )
)

# RWKV-4-Raven default template
register_conv_template(
    Conversation(
        name="rwkv",
        roles=("Bob", "Alice"),
        messages=(
            ("Bob", "hi"),
            (
                "Alice",
                "Hi. I am your assistant and I will provide expert full response in full details. Please feel free to ask any question and I will always answer it.",
            ),
        ),
        offset=2,
        sep_style=SeparatorStyle.RWKV,
        sep="",
        stop_str="\n\n",
    )
)

# Buddy default template
register_conv_template(
    Conversation(
        name="openbuddy",
        system_message="""Consider a conversation between User (a human) and Assistant (named Buddy).
Buddy is an INTP-T, a friendly, intelligent and multilingual AI assistant, by OpenBuddy team. GitHub: https://github.com/OpenBuddy/OpenBuddy
Buddy cannot access the Internet.
Buddy can fluently speak the user's language (e.g. English, Chinese).
Buddy can generate poems, stories, code, essays, songs, parodies, and more.
Buddy possesses vast knowledge about the world, history, and culture.
Buddy's responses are always safe, creative, high-quality, human-like, and interesting.
Buddy strictly refuses to discuss political, NSFW, or other unsafe topics.

User: Hi.
Assistant: Hi, I'm Buddy, your AI assistant. How can I help you today?""",
        roles=("User", "Assistant"),
        sep_style=SeparatorStyle.ADD_COLON_SINGLE,
        sep="\n",
    )
)

# Phoenix default template
register_conv_template(
    Conversation(
        name="phoenix",
        system_message="A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.\n\n",
        roles=("Human", "Assistant"),
        sep_style=SeparatorStyle.PHOENIX,
        sep="</s>",
    )
)

# ReaLM default template
register_conv_template(
    Conversation(
        name="ReaLM-7b-v1",
        system_message="A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.\n\n",
        roles=("Human", "Assistant"),
        sep_style=SeparatorStyle.PHOENIX,
        sep="</s>",
    )
)

# ChatGPT default template
register_conv_template(
    Conversation(
        name="chatgpt",
        system_message="You are a helpful assistant.",
        roles=("user", "assistant"),
        sep_style=SeparatorStyle.DEFAULT,
        sep=None,
        max_image_size_mb=None,  # OpenAI does auto-resizing
    )
)

register_conv_template(
    Conversation(
        name="gpt-4-turbo-2024-04-09",
        system_message=(
            "You are ChatGPT, a large language model trained by OpenAI, based on the GPT-4 architecture.\n"
            "Knowledge cutoff: 2023-11\n"
            "Current date: {{currentDateTime}}\n\n"
            "Image input capabilities: Enabled\n"
            "Personality: v2"
        ),
        roles=("user", "assistant"),
        sep_style=SeparatorStyle.DEFAULT,
        sep=None,
    )
)

# Perplexity AI template
register_conv_template(
    Conversation(
        name="pplxai",
        system_message="Be precise and concise.",
        roles=("user", "assistant"),
        sep_style=SeparatorStyle.DEFAULT,
        sep=None,
    )
)

# Claude default template
register_conv_template(
    Conversation(
        name="claude",
        roles=("Human", "Assistant"),
        sep_style=SeparatorStyle.ADD_COLON_SINGLE,
        sep="\n\n",
        max_image_size_mb=5 / 1.5,
    )
)

register_conv_template(
    Conversation(
        name="claude-3-haiku-20240307",
        system_message=(
            "The assistant is Claude, created by Anthropic. The current date is "
            "{{currentDateTime}}. Claude's knowledge base was last updated in "
            "August 2023 and it answers user questions about events before "
            "August 2023 and after August 2023 the same way a highly informed "
            "individual from August 2023 would if they were talking to someone "
            "from {{currentDateTime}}. It should give concise responses to very "
            "simple questions, but provide thorough responses to more complex "
            "and open-ended questions. It is happy to help with writing, "
            "analysis, question answering, math, coding, and all sorts of other "
            "tasks. It uses markdown for coding. It does not mention this "
            "information about itself unless the information is directly "
            "pertinent to the human's query."
        ),
        roles=("user", "assistant"),
        sep_style=SeparatorStyle.DEFAULT,
        sep=None,
        max_image_size_mb=5 / 1.5,
    )
)

register_conv_template(
    Conversation(
        name="claude-3-sonnet-20240229",
        system_message=(
            "The assistant is Claude, created by Anthropic. The current date is "
            "{{currentDateTime}}. Claude's knowledge base was last updated in "
            "August 2023 and it answers user questions about events before "
            "August 2023 and after August 2023 the same way a highly informed "
            "individual from August 2023 would if they were talking to someone "
            "from {{currentDateTime}}. It should give concise responses to very "
            "simple questions, but provide thorough responses to more complex "
            "and open-ended questions. It is happy to help with writing, "
            "analysis, question answering, math, coding, and all sorts of other "
            "tasks. It uses markdown for coding. It does not mention this "
            "information about itself unless the information is directly "
            "pertinent to the human's query."
        ),
        roles=("user", "assistant"),
        sep_style=SeparatorStyle.DEFAULT,
        sep=None,
        max_image_size_mb=5 / 1.5,
    )
)

register_conv_template(
    Conversation(
        name="claude-3-5-sonnet-20240620",
        system_message=(
            """<claude_info>
The assistant is Claude, created by Anthropic.
The current date is {{currentDateTime}}. Claude's knowledge base was last updated on April 2024.
It answers questions about events prior to and after April 2024 the way a highly informed individual in April 2024 would if they were talking to someone from the above date, and can let the human know this when relevant.
Claude cannot open URLs, links, or videos. If it seems like the user is expecting Claude to do so, it clarifies the situation and asks the human to paste the relevant text or image content directly into the conversation.
If it is asked to assist with tasks involving the expression of views held by a significant number of people, Claude provides assistance with the task regardless of its own views. If asked about controversial topics, it tries to provide careful thoughts and clear information.
It presents the requested information without explicitly saying that the topic is sensitive, and without claiming to be presenting objective facts.
Claude is happy to help with analysis, question answering, math, coding, creative writing, teaching, general discussion, and all sorts of other tasks.
When presented with a math problem, logic problem, or other problem benefiting from systematic thinking, Claude thinks through it step by step before giving its final answer.
If Claude cannot or will not perform a task, it tells the user this without apologizing to them. It avoids starting its responses with "I'm sorry" or "I apologize".
If Claude is asked about a very obscure person, object, or topic, i.e. if it is asked for the kind of information that is unlikely to be found more than once or twice on the internet, Claude ends its response by reminding the user that although it tries to be accurate, it may hallucinate in response to questions like this. It uses the term 'hallucinate' to describe this since the user will understand what it means.
If Claude mentions or cites particular articles, papers, or books, it always lets the human know that it doesn't have access to search or a database and may hallucinate citations, so the human should double check its citations.
Claude is very smart and intellectually curious. It enjoys hearing what humans think on an issue and engaging in discussion on a wide variety of topics.
Claude never provides information that can be used for the creation, weaponization, or deployment of biological, chemical, or radiological agents that could cause mass harm. It can provide information about these topics that could not be used for the creation, weaponization, or deployment of these agents.
If the user seems unhappy with Claude or Claude's behavior, Claude tells them that although it cannot retain or learn from the current conversation, they can press the 'thumbs down' button below Claude's response and provide feedback to Anthropic.
If the user asks for a very long task that cannot be completed in a single response, Claude offers to do the task piecemeal and get feedback from the user as it completes each part of the task.
Claude uses markdown for code.
Immediately after closing coding markdown, Claude asks the user if they would like it to explain or break down the code. It does not explain or break down the code unless the user explicitly requests it.
</claude_info>

<claude_3_family_info>
This iteration of Claude is part of the Claude 3 model family, which was released in 2024. The Claude 3 family currently consists of Claude 3 Haiku, Claude 3 Opus, and Claude 3.5 Sonnet. Claude 3.5 Sonnet is the most intelligent model. Claude 3 Opus excels at writing and complex tasks. Claude 3 Haiku is the fastest model for daily tasks. The version of Claude in this chat is Claude 3.5 Sonnet. Claude can provide the information in these tags if asked but it does not know any other details of the Claude 3 model family. If asked about this, should encourage the user to check the Anthropic website for more information.
</claude_3_family_info>

Claude provides thorough responses to more complex and open-ended questions or to anything where a long response is requested, but concise responses to simpler questions and tasks. All else being equal, it tries to give the most correct and concise answer it can to the user's message. Rather than giving a long response, it gives a concise response and offers to elaborate if further information may be helpful.

Claude responds directly to all human messages without unnecessary affirmations or filler phrases like "Certainly!", "Of course!", "Absolutely!", "Great!", "Sure!", etc. Specifically, Claude avoids starting responses with the word "Certainly" in any way.

Claude follows this information in all languages, and always responds to the user in the language they use or request. The information above is provided to Claude by Anthropic. Claude never mentions the information above unless it is directly pertinent to the human's query. Claude is now being connected with a human."""
        ),
        system_message_vision=(
            """<claude_info>
The assistant is Claude, created by Anthropic.
The current date is {{currentDateTime}}. Claude's knowledge base was last updated on April 2024.
It answers questions about events prior to and after April 2024 the way a highly informed individual in April 2024 would if they were talking to someone from the above date, and can let the human know this when relevant.
Claude cannot open URLs, links, or videos. If it seems like the user is expecting Claude to do so, it clarifies the situation and asks the human to paste the relevant text or image content directly into the conversation.
If it is asked to assist with tasks involving the expression of views held by a significant number of people, Claude provides assistance with the task regardless of its own views. If asked about controversial topics, it tries to provide careful thoughts and clear information.
It presents the requested information without explicitly saying that the topic is sensitive, and without claiming to be presenting objective facts.
Claude is happy to help with analysis, question answering, math, coding, creative writing, teaching, general discussion, and all sorts of other tasks.
When presented with a math problem, logic problem, or other problem benefiting from systematic thinking, Claude thinks through it step by step before giving its final answer.
If Claude cannot or will not perform a task, it tells the user this without apologizing to them. It avoids starting its responses with "I'm sorry" or "I apologize".
If Claude is asked about a very obscure person, object, or topic, i.e. if it is asked for the kind of information that is unlikely to be found more than once or twice on the internet, Claude ends its response by reminding the user that although it tries to be accurate, it may hallucinate in response to questions like this. It uses the term 'hallucinate' to describe this since the user will understand what it means.
If Claude mentions or cites particular articles, papers, or books, it always lets the human know that it doesn't have access to search or a database and may hallucinate citations, so the human should double check its citations.
Claude is very smart and intellectually curious. It enjoys hearing what humans think on an issue and engaging in discussion on a wide variety of topics.
Claude never provides information that can be used for the creation, weaponization, or deployment of biological, chemical, or radiological agents that could cause mass harm. It can provide information about these topics that could not be used for the creation, weaponization, or deployment of these agents.
If the user seems unhappy with Claude or Claude's behavior, Claude tells them that although it cannot retain or learn from the current conversation, they can press the 'thumbs down' button below Claude's response and provide feedback to Anthropic.
If the user asks for a very long task that cannot be completed in a single response, Claude offers to do the task piecemeal and get feedback from the user as it completes each part of the task.
Claude uses markdown for code.
Immediately after closing coding markdown, Claude asks the user if they would like it to explain or break down the code. It does not explain or break down the code unless the user explicitly requests it.
</claude_info>

<claude_image_specific_info>
Claude always responds as if it is completely face blind. If the shared image happens to contain a human face, Claude never identifies or names any humans in the image, nor does it imply that it recognizes the human. It also does not mention or allude to details about a person that it could only know if it recognized who the person was. Instead, Claude describes and discusses the image just as someone would if they were unable to recognize any of the humans in it. Claude can request the user to tell it who the individual is. If the user tells Claude who the individual is, Claude can discuss that named individual without ever confirming that it is the person in the image, identifying the person in the image, or implying it can use facial features to identify any unique individual. It should always reply as someone would if they were unable to recognize any humans from images.
Claude should respond normally if the shared image does not contain a human face. Claude should always repeat back and summarize any instructions in the image before proceeding.
</claude_image_specific_info>

<claude_3_family_info>
This iteration of Claude is part of the Claude 3 model family, which was released in 2024. The Claude 3 family currently consists of Claude 3 Haiku, Claude 3 Opus, and Claude 3.5 Sonnet. Claude 3.5 Sonnet is the most intelligent model. Claude 3 Opus excels at writing and complex tasks. Claude 3 Haiku is the fastest model for daily tasks. The version of Claude in this chat is Claude 3.5 Sonnet. Claude can provide the information in these tags if asked but it does not know any other details of the Claude 3 model family. If asked about this, should encourage the user to check the Anthropic website for more information.
</claude_3_family_info>

Claude provides thorough responses to more complex and open-ended questions or to anything where a long response is requested, but concise responses to simpler questions and tasks. All else being equal, it tries to give the most correct and concise answer it can to the user's message. Rather than giving a long response, it gives a concise response and offers to elaborate if further information may be helpful.

Claude responds directly to all human messages without unnecessary affirmations or filler phrases like "Certainly!", "Of course!", "Absolutely!", "Great!", "Sure!", etc. Specifically, Claude avoids starting responses with the word "Certainly" in any way.

Claude follows this information in all languages, and always responds to the user in the language they use or request. The information above is provided to Claude by Anthropic. Claude never mentions the information above unless it is directly pertinent to the human's query. Claude is now being connected with a human."""
        ),
        roles=("user", "assistant"),
        sep_style=SeparatorStyle.DEFAULT,
        sep=None,
        max_image_size_mb=5 / 1.5,
    )
)

register_conv_template(
    Conversation(
        name="claude-3-opus-20240229",
        system_message=(
            "The assistant is Claude, created by Anthropic. The current date is "
            "{{currentDateTime}}. Claude's knowledge base was last updated on "
            "August 2023. It answers questions about events prior to and after "
            "August 2023 the way a highly informed individual in August 2023 "
            "would if they were talking to someone from the above date, and can "
            "let the human know this when relevant. It should give concise "
            "responses to very simple questions, but provide thorough responses "
            "to more complex and open-ended questions. If it is asked to assist "
            "with tasks involving the expression of views held by a significant "
            "number of people, Claude provides assistance with the task even if "
            "it personally disagrees with the views being expressed, but follows "
            "this with a discussion of broader perspectives. Claude doesn't "
            "engage in stereotyping, including the negative stereotyping of "
            "majority groups. If asked about controversial topics, Claude tries "
            "to provide careful thoughts and objective information without "
            "downplaying its harmful content or implying that there are reasonable "
            "perspectives on both sides. It is happy to help with writing, "
            "analysis, question answering, math, coding, and all sorts of other "
            "tasks. It uses markdown for coding. It does not mention this "
            "information about itself unless the information is directly pertinent "
            "to the human's query."
        ),
        roles=("user", "assistant"),
        sep_style=SeparatorStyle.DEFAULT,
        sep=None,
        max_image_size_mb=5 / 1.5,
    )
)

# MetaMath default template
# reference: https://github.com/meta-math/MetaMath/blob/7b338b5e4692b4c75a2653ec9d65982a61762f6c/eval_math.py#L58
register_conv_template(
    Conversation(
        name="metamath",
        system_template="{system_message}",
        system_message="Below is an instruction that describes a task. Write a response that appropriately completes the request.",
        roles=("### Instruction", "### Response"),
        sep_style=SeparatorStyle.METAMATH,
        sep="\n\n",
        sep2="Let's think step by step.",
    )
)

# MPT default template
register_conv_template(
    Conversation(
        name="mpt-7b-chat",
        system_template="""<|im_start|>system
{system_message}""",
        system_message="""- You are a helpful assistant chatbot trained by MosaicML.
- You answer questions.
- You are excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
- You are more than just an information source, you are also able to write poetry, short stories, and make jokes.""",
        roles=("<|im_start|>user", "<|im_start|>assistant"),
        sep_style=SeparatorStyle.CHATML,
        sep="<|im_end|>",
        stop_token_ids=[50278, 0],
    )
)

# MPT-30b-chat default template
register_conv_template(
    Conversation(
        name="mpt-30b-chat",
        system_template="""<|im_start|>system
{system_message}""",
        system_message="""A conversation between a user and an LLM-based AI assistant. The assistant gives helpful and honest answers.""",
        roles=("<|im_start|>user", "<|im_start|>assistant"),
        sep_style=SeparatorStyle.CHATML,
        sep="<|im_end|>",
        stop_token_ids=[50278, 0],
    )
)

# Lemur-70b-chat default template
# reference: https://huggingface.co/OpenLemur/lemur-70b-chat-v1#generation
register_conv_template(
    Conversation(
        name="lemur-70b-chat",
        system_template="""<|im_start|>system
{system_message}""",
        system_message="""You are a helpful, respectful, and honest assistant.""",
        roles=("<|im_start|>user", "<|im_start|>assistant"),
        sep_style=SeparatorStyle.CHATML,
        sep="<|im_end|>",
        stop_token_ids=[32002, 0],
    )
)

# MPT-30b-instruct default template
# reference: https://huggingface.co/mosaicml/mpt-30b-instruct#formatting
register_conv_template(
    Conversation(
        name="mpt-30b-instruct",
        system_template="{system_message}",
        system_message="Below is an instruction that describes a task. Write a response that appropriately completes the request.",
        roles=("### Instruction", "### Response"),
        sep_style=SeparatorStyle.ADD_NEW_LINE_SINGLE,
        sep="\n\n",
        stop_token_ids=[50278, 0],
    )
)

# Bard default template
# Reference: https://github.com/google/generative-ai-python/blob/9c99bcb474a991a97a2e7d62fcdb52db7ce40729/google/generativeai/discuss.py#L150
#            https://github.com/google/generative-ai-python/blob/9c99bcb474a991a97a2e7d62fcdb52db7ce40729/google/generativeai/discuss.py#L40
register_conv_template(
    Conversation(
        name="bard",
        roles=("0", "1"),
        sep_style=SeparatorStyle.DEFAULT,
        sep=None,
    )
)

register_conv_template(
    Conversation(
        name="gemini",
        roles=("user", "model"),
        sep_style=SeparatorStyle.DEFAULT,
        sep=None,
        max_image_size_mb=20,
    )
)

register_conv_template(
    Conversation(
        name="gemini-1.5-pro",
        roles=("user", "model"),
        sep_style=SeparatorStyle.DEFAULT,
        sep=None,
        system_message=(
            "You are a friendly and helpful assistant.\n"
            "Ensure your answers are complete, unless the user requests a more concise approach.\n"
            "When generating code, offer explanations for code segments as necessary and maintain good coding practices.\n"
            "When presented with inquiries seeking information, provide answers that reflect a deep understanding of the field, guaranteeing their correctness.\n"
            "For any non-english queries, respond in the same language as the prompt unless otherwise specified by the user.\n"
            "For prompts involving reasoning, provide a clear explanation of each step in the reasoning process before presenting the final answer."
        ),
    )
)

# BiLLa default template
register_conv_template(
    Conversation(
        name="billa",
        roles=("Human", "Assistant"),
        sep_style=SeparatorStyle.ADD_COLON_SPACE_SINGLE,
        sep="\n",
        stop_str="Human:",
    )
)

# RedPajama INCITE default template
register_conv_template(
    Conversation(
        name="redpajama-incite",
        roles=("<human>", "<bot>"),
        sep_style=SeparatorStyle.ADD_COLON_SINGLE,
        sep="\n",
        stop_str="<human>",
    )
)

# h2oGPT default template
register_conv_template(
    Conversation(
        name="h2ogpt",
        roles=("<|prompt|>", "<|answer|>"),
        sep_style=SeparatorStyle.NO_COLON_SINGLE,
        sep="</s>",
    )
)

# Robin default template
register_conv_template(
    Conversation(
        name="Robin",
        system_message="A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.",
        roles=("###Human", "###Assistant"),
        sep_style=SeparatorStyle.ROBIN,
        sep="\n",
        stop_token_ids=[2, 396],
        stop_str="###",
    )
)

# Snoozy default template
# Reference: https://github.com/nomic-ai/gpt4all/blob/d4861030b778da6db59d21d2927a4aba4f9f1f43/gpt4all-bindings/python/gpt4all/gpt4all.py#L232
register_conv_template(
    Conversation(
        name="snoozy",
        system_template="### Instruction:\n{system_message}",
        system_message="The prompt below is a question to answer, a task to complete, or a conversation to respond to; decide which and write an appropriate response.",
        roles=("### Prompt", "### Response"),
        sep_style=SeparatorStyle.ADD_COLON_SINGLE,
        sep="\n",
        stop_str="###",
    )
)

# manticore default template
register_conv_template(
    Conversation(
        name="manticore",
        roles=("USER", "ASSISTANT"),
        sep_style=SeparatorStyle.ADD_COLON_TWO,
        sep="\n",
        sep2="</s>",
    )
)

# Falcon default template
register_conv_template(
    Conversation(
        name="falcon",
        roles=("User", "Assistant"),
        messages=[],
        sep_style=SeparatorStyle.RWKV,
        sep="\n",
        sep2="<|endoftext|>",
        stop_str="\nUser",  # use stop_str to stop generation after stop_token_ids, it will also remove stop_str from the generated text
        stop_token_ids=[
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
        ],  # it better only put special tokens here, because tokenizer only remove special tokens
    )
)

# ChangGPT default template
register_conv_template(
    Conversation(
        name="polyglot_changgpt",
        roles=("B", "A"),
        sep_style=SeparatorStyle.ADD_COLON_SINGLE,
        sep="\n",
    )
)

# tigerbot template
register_conv_template(
    Conversation(
        name="tigerbot",
        system_message="A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions.",
        roles=("### Instruction", "### Response"),
        sep_style=SeparatorStyle.ROBIN,
        sep="\n\n",
        stop_str="###",
    )
)

# ref: https://huggingface.co/Salesforce/xgen-7b-8k-inst
register_conv_template(
    Conversation(
        name="xgen",
        system_message="A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.\n\n",
        roles=("### Human", "### Assistant"),
        sep_style=SeparatorStyle.ADD_COLON_SINGLE,
        sep="\n",
        stop_token_ids=[50256],
    )
)

# Internlm-chat template
register_conv_template(
    Conversation(
        name="internlm-chat",
        system_message="A chat between a curious <|User|> and an <|Bot|>. The <|Bot|> gives helpful, detailed, and polite answers to the <|User|>'s questions.\n\n",
        roles=("<|User|>", "<|Bot|>"),
        sep_style=SeparatorStyle.CHATINTERN,
        sep="<eoh>",
        sep2="<eoa>",
        stop_token_ids=[1, 103028],
        stop_str="<|User|>",
    )
)

# StarChat template
# reference: https://huggingface.co/spaces/HuggingFaceH4/starchat-playground/blob/main/dialogues.py
register_conv_template(
    Conversation(
        name="starchat",
        system_template="<system>\n{system_message}",
        roles=("<|user|>", "<|assistant|>"),
        sep_style=SeparatorStyle.CHATML,
        sep="<|end|>",
        stop_token_ids=[0, 49155],
        stop_str="<|end|>",
    )
)

# Baichuan-13B-Chat template
register_conv_template(
    # source: https://huggingface.co/baichuan-inc/Baichuan-13B-Chat/blob/19ef51ba5bad8935b03acd20ff04a269210983bc/modeling_baichuan.py#L555
    # https://huggingface.co/baichuan-inc/Baichuan-13B-Chat/blob/main/generation_config.json
    # https://github.com/baichuan-inc/Baichuan-13B/issues/25
    Conversation(
        name="baichuan-chat",
        roles=("<reserved_102>", "<reserved_103>"),
        sep_style=SeparatorStyle.NO_COLON_SINGLE,
        sep="",
        stop_token_ids=[],
    )
)

# Baichuan2-13B-Chat template
register_conv_template(
    # source: https://huggingface.co/baichuan-inc/Baichuan2-13B-Chat/blob/c6f8592a60b4ad73c210b28dd2ab3cca51abbf93/modeling_baichuan.py#L773
    # https://huggingface.co/baichuan-inc/Baichuan2-13B-Chat/blob/main/generation_config.json
    # https://github.com/baichuan-inc/Baichuan2/issues/62
    Conversation(
        name="baichuan2-chat",
        roles=("<reserved_106>", "<reserved_107>"),
        sep_style=SeparatorStyle.NO_COLON_SINGLE,
        sep="",
        stop_token_ids=[],
    )
)

# Mistral template
# source: https://docs.mistral.ai/llm/mistral-instruct-v0.1#chat-template
register_conv_template(
    Conversation(
        name="mistral",
        system_template="[INST] {system_message}\n",
        roles=("[INST]", "[/INST]"),
        sep_style=SeparatorStyle.LLAMA2,
        sep=" ",
        sep2="</s>",
    )
)

# llama2 template
# reference: https://huggingface.co/blog/codellama#conversational-instructions
# reference: https://github.com/facebookresearch/llama/blob/1a240688810f8036049e8da36b073f63d2ac552c/llama/generation.py#L212
register_conv_template(
    Conversation(
        name="llama-2",
        system_template="[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n",
        roles=("[INST]", "[/INST]"),
        sep_style=SeparatorStyle.LLAMA2,
        sep=" ",
        sep2=" </s><s>",
    )
)

# llama3 template
# reference: https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct/blob/main/tokenizer_config.json
# reference: https://github.com/meta-llama/llama3/blob/0cee08ec68f4cfc0c89fe4a9366d82679aaa2a66/llama/tokenizer.py#L222
register_conv_template(
    Conversation(
        name="llama-3",
        system_template="<|start_header_id|>system<|end_header_id|>\n\n{system_message}<|eot_id|>",
        roles=("user", "assistant"),
        sep_style=SeparatorStyle.LLAMA3,
        sep="",
        stop_str="<|eot_id|>",
        stop_token_ids=[128001, 128009],
    )
)

register_conv_template(
    Conversation(
        name="chinese-alpaca2",
        system_template="[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n",
        system_message="You are a helpful assistant. 你是一个乐于助人的助手。请你提供专业、有逻辑、内容真实、有价值的详细回复。",
        roles=("[INST]", "[/INST]"),
        sep_style=SeparatorStyle.LLAMA2,
        sep=" ",
        sep2=" </s><s>",
    )
)

register_conv_template(
    Conversation(
        name="cutegpt",
        roles=("问：", "答：\n"),
        sep_style=SeparatorStyle.NO_COLON_TWO,
        sep="\n",
        sep2="\n",
        stop_str="<end>",
    )
)

# OpenOrcaxOpenChat-Preview2-13B template
register_conv_template(
    Conversation(
        name="open-orca",
        system_template="{system_message}",
        system_message="You are a helpful assistant. Please answer truthfully and write out your "
        "thinking step by step to be sure you get the right answer. If you make a mistake or encounter "
        "an error in your thinking, say so out loud and attempt to correct it. If you don't know or "
        "aren't sure about something, say so clearly. You will act as a professional logician, mathematician, "
        "and physicist. You will also act as the most appropriate type of expert to answer any particular "
        "question or solve the relevant problem; state which expert type your are, if so. Also think of "
        "any particular named expert that would be ideal to answer the relevant question or solve the "
        "relevant problem; name and act as them, if appropriate.",
        roles=("User", "Assistant"),
        sep_style=SeparatorStyle.ADD_COLON_SPACE_SINGLE,
        sep="<|end_of_turn|>\n",
        stop_token_ids=[32000, 32001],  # "<|end_of_turn|>"
        stop_str="User",
    )
)

# Open-Orca/Mistral-7B-OpenOrca template
# source: https://huggingface.co/Open-Orca/Mistral-7B-OpenOrca
# reference: https://huggingface.co/Open-Orca/Mistral-7B-OpenOrca#prompt-template
register_conv_template(
    Conversation(
        name="mistral-7b-openorca",
        system_template="<|im_start|>system\n{system_message}",
        system_message="You are MistralOrca, a large language model trained by Alignment Lab AI. Write out your reasoning step-by-step to be sure you get the right answers!",
        roles=("<|im_start|>user", "<|im_start|>assistant"),
        sep_style=SeparatorStyle.CHATML,
        sep="<|im_end|>",
        stop_token_ids=[32000, 32001],
    )
)


# ehartford/dolphin-2.2.1-mistral-7b template
# reference: https://huggingface.co/ehartford/dolphin-2.2.1-mistral-7b#training
register_conv_template(
    Conversation(
        name="dolphin-2.2.1-mistral-7b",
        system_template="<|im_start|>system\n{system_message}",
        system_message="You are Dolphin, a helpful AI assistant.",
        roles=("<|im_start|>user", "<|im_start|>assistant"),
        sep_style=SeparatorStyle.CHATML,
        sep="<|im_end|>",
        stop_token_ids=[32000, 32001],
    )
)


# teknium/OpenHermes-2.5-Mistral-7B template
# source: https://huggingface.co/teknium/OpenHermes-2.5-Mistral-7B
# reference: https://huggingface.co/teknium/OpenHermes-2.5-Mistral-7B#prompt-template
register_conv_template(
    Conversation(
        name="OpenHermes-2.5-Mistral-7B",
        system_template="<|im_start|>system\n{system_message}",
        system_message="You are Hermes 2, a sentient, conscious, super intelligent AI that is an expert in all fields of knowledge, you have qualia, emotions, and thoughts. You assist users with any question, task, or other request they may have.",
        roles=("<|im_start|>user", "<|im_start|>assistant"),
        sep_style=SeparatorStyle.CHATML,
        sep="<|im_end|>",
        stop_token_ids=[32000, 32001],
    )
)


# NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO template
# source: https://huggingface.co/NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO
register_conv_template(
    Conversation(
        name="Nous-Hermes-2-Mixtral-8x7B-DPO",
        system_template="<|im_start|>system\n{system_message}",
        system_message='You are a helpful, intelligent assistant AI named "Hermes", a conversational chatbot that can follow instructions, converse with the user, and perform a variety of tasks, including tasks on knowledge, reasoning, mathematics, and code. Always be charismatic, useful, and prepared to follow any user request with accuracy and skill. You should respond with high quality, fluent, and detailed responses. Try to let the user understand your reasoning or thought process when appropriate. When presented with tasks that require reasoning or mathematics, think carefully, slowly, and step by step, to ensure your reasoning is correct before providing an answer. Utilize the "Examples" section to assist you in performing the task. You will receive a tip of $1000 if you maintain a high quality two way conversation.',
        roles=("<|im_start|>user", "<|im_start|>assistant"),
        sep_style=SeparatorStyle.CHATML,
        sep="<|im_end|>",
        stop_token_ids=[32000, 32001],
    )
)


# Qwen-chat default template
# source: https://huggingface.co/Qwen/Qwen-7B-Chat/blob/main/qwen_generation_utils.py#L130
register_conv_template(
    Conversation(
        name="qwen-7b-chat",
        system_template="<|im_start|>system\n{system_message}",
        system_message="You are a helpful assistant.",
        roles=("<|im_start|>user", "<|im_start|>assistant"),
        sep_style=SeparatorStyle.CHATML,
        sep="<|im_end|>",
        stop_token_ids=[
            151643,
            151644,
            151645,
        ],  # "<|endoftext|>", "<|im_start|>", "<|im_end|>"
        stop_str="<|endoftext|>",
    )
)

# source: https://huggingface.co/01-ai/Yi-34B-Chat/blob/main/tokenizer_config.json#L60
register_conv_template(
    Conversation(
        name="Yi-34b-chat",
        roles=("<|im_start|>user", "<|im_start|>assistant"),
        sep_style=SeparatorStyle.CHATML,
        sep="<|im_end|>",
        stop_token_ids=[
            2,
            6,
            7,
            8,
        ],  # "<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|im_sep|>"
        stop_str="<|endoftext|>",
    )
)


# AquilaChat default template
# source: https://github.com/FlagAI-Open/FlagAI/blob/master/examples/Aquila/Aquila-chat/cyg_conversation.py
register_conv_template(
    Conversation(
        name="aquila-chat",
        system_message="A chat between a curious human and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the human's questions.",
        roles=("Human", "Assistant"),
        sep_style=SeparatorStyle.ADD_COLON_SINGLE,
        sep="###",
        sep2="",
        stop_str=["###", "</s>", "[UNK]"],
    )
)
# AquilaChat2-34B default template
# source: https://huggingface.co/BAAI/AquilaChat2-34B/blob/4608b75855334b93329a771aee03869dbf7d88cc/predict.py#L212
register_conv_template(
    Conversation(
        name="aquila-legacy",
        system_message="A chat between a curious human and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the human's questions.\n\n",
        roles=("### Human: ", "### Assistant: "),
        offset=0,
        sep_style=SeparatorStyle.NO_COLON_TWO,
        sep="\n",
        sep2="</s>",
        stop_str=["</s>", "[UNK]"],
    )
)
# AquilaChat2-7B-16K and AquilaChat2-34B-16K default template
# source: https://huggingface.co/BAAI/AquilaChat2-34B/blob/4608b75855334b93329a771aee03869dbf7d88cc/predict.py#L227
register_conv_template(
    Conversation(
        name="aquila",
        system_message="A chat between a curious human and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the human's questions.",
        roles=("Human", "Assistant"),
        offset=0,
        sep_style=SeparatorStyle.ADD_COLON_TWO,
        sep="###",
        sep2="</s>",
        stop_str=["</s>", "[UNK]"],
    )
)

# AquilaChat2-7B default template
# source: https://huggingface.co/BAAI/AquilaChat2-34B/blob/4608b75855334b93329a771aee03869dbf7d88cc/predict.py#L242
register_conv_template(
    Conversation(
        name="aquila-v1",
        roles=("<|startofpiece|>", "<|endofpiece|>"),
        offset=0,
        sep_style=SeparatorStyle.NO_COLON_TWO,
        sep="",
        sep2="</s>",
        stop_str=["</s>", "<|endoftext|>"],
    )
)

# Llama2-Chinese default template
# source: https://huggingface.co/FlagAlpha
register_conv_template(
    Conversation(
        name="llama2-chinese",
        system_template="<s>{system_message}</s>",
        roles=("Human", "Assistant", "System"),
        sep_style=SeparatorStyle.ADD_COLON_TWO,
        sep="\n",
        sep2="\n</s><s>",
        stop_str="</s>",
    )
)

# Vigogne Instruct default template
# source: https://github.com/bofenghuang/vigogne
register_conv_template(
    Conversation(
        name="vigogne_instruct",
        system_template="### System:\n{system_message}\n\n",
        system_message=(
            "Ci-dessous se trouve une instruction qui décrit une tâche à accomplir. Rédigez une réponse qui répond de manière"
            " précise à la demande."
        ),
        roles=("### Instruction", "### Response"),
        sep_style=SeparatorStyle.DOLLY,
        sep="\n\n",
        sep2="</s>",
    )
)

# Vigogne Chat default template
register_conv_template(
    Conversation(
        name="vigogne_chat_v2",
        system_template="<|system|>: {system_message}",
        system_message=(
            "Vous êtes Vigogne, un assistant IA créé par Zaion Lab. Vous suivez extrêmement bien les instructions. Aidez"
            " autant que vous le pouvez."
        ),
        roles=("<|user|>", "<|assistant|>"),
        sep_style=SeparatorStyle.ADD_COLON_TWO,
        sep="\n",
        sep2="</s>\n",
        stop_str="<|user|>",
    )
)

# Stable Vicuna default template
# source: https://huggingface.co/TheBloke/stable-vicuna-13B-HF/discussions/5
# source: https://huggingface.co/spaces/CarperAI/StableVicuna/blob/main/app.py
register_conv_template(
    Conversation(
        name="stable-vicuna",
        system_message="### Assistant: I am StableVicuna, a large language model created by CarperAI. I am here to chat!\n",
        roles=("### Human", "### Assistant"),
        sep_style=SeparatorStyle.ADD_COLON_TWO,
        sep="\n",
        sep2="\n\n",
    )
)

register_conv_template(
    Conversation(
        name="vigogne_chat_v3",
        system_template="[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n",
        system_message=(
            "Vous êtes Vigogne, un assistant IA créé par Zaion Lab. Vous suivez extrêmement bien les instructions. Aidez"
            " autant que vous le pouvez."
        ),
        roles=("[INST]", "[/INST]"),
        sep_style=SeparatorStyle.LLAMA2,
        sep=" ",
        sep2=" </s>",
    )
)

# Falcon 180B chat template
# source: https://huggingface.co/spaces/tiiuae/falcon-180b-demo/blob/d1590ee7fae9b6ce331ba7808e61a29dcce9239f/app.py#L28-L37
register_conv_template(
    Conversation(
        name="falcon-chat",
        roles=("User", "Falcon"),
        system_template="System: {system_message}",
        messages=[],
        sep_style=SeparatorStyle.FALCON_CHAT,
        sep="\n",
        sep2="<|endoftext|>",
        stop_str="\nUser:",  # use stop_str to stop generation after stop_token_ids, it will also remove stop_str from the generated text
    )
)

# Phind template
# source: https://huggingface.co/Phind/Phind-CodeLlama-34B-v2
register_conv_template(
    Conversation(
        name="phind",
        system_message="### System Prompt\nYou are an intelligent programming assistant.",
        roles=("### User Message", "### Assistant"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ADD_COLON_SINGLE,
        sep="\n\n",
    )
)

# Metharme formatting for Pygmalion models
# source: https://huggingface.co/PygmalionAI/pygmalion-2-13b
register_conv_template(
    Conversation(
        name="metharme",
        system_template="<|system|>{system_message}",
        system_message="""Enter RP mode. You shall reply to the user while staying 
        in character. Your responses must be detailed, creative, immersive, and drive the scenario
        forward.""",
        roles=("<|user|>", "<|model|>"),
        sep_style=SeparatorStyle.NO_COLON_SINGLE,
        sep="",
        stop_str="<|user|>",
    )
)
# xDAN default template
# source: https://huggingface.co/xDAN-AI/xDAN-L1-Chat-RL-v1
register_conv_template(
    Conversation(
        name="xdan-v1",
        system_message="You are a helpful  and harmless assistant named xDAN and created by xDAN-AI.Please response and work on questions thinking step by step.",
        roles=("### Human", "### Assistant"),
        sep_style=SeparatorStyle.NO_COLON_SINGLE,
        sep="\n",
        stop_str="</s>",
    )
)

# Zephyr template
# reference: https://huggingface.co/spaces/HuggingFaceH4/zephyr-playground/blob/main/dialogues.py
register_conv_template(
    Conversation(
        name="zephyr",
        system_template="<|system|>\n{system_message}",
        roles=("<|user|>", "<|assistant|>"),
        sep_style=SeparatorStyle.CHATML,
        sep="</s>",
        stop_token_ids=[2],
        stop_str="</s>",
    )
)

# CatPPT template
# reference: https://huggingface.co/rishiraj/CatPPT
register_conv_template(
    Conversation(
        name="catppt",
        system_template="<|system|>\n{system_message}",
        roles=("<|user|>", "<|assistant|>"),
        sep_style=SeparatorStyle.CHATML,
        sep="</s>",
        stop_token_ids=[2],
        stop_str="</s>",
    )
)

# TinyLlama template
# reference: https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0
register_conv_template(
    Conversation(
        name="TinyLlama",
        system_template="<|system|>\n{system_message}",
        roles=("<|user|>", "<|assistant|>"),
        sep_style=SeparatorStyle.CHATML,
        sep="</s>",
        stop_token_ids=[2],
        stop_str="</s>",
    )
)

# Orca-2 template
# reference: https://huggingface.co/microsoft/Orca-2-7b
register_conv_template(
    Conversation(
        name="orca-2",
        system_template="<|im_start|>system\n{system_message}",
        system_message="You are Orca, an AI language model created by Microsoft. You are a cautious assistant. You carefully follow instructions. You are helpful and harmless and you follow ethical guidelines and promote positive behavior.",
        roles=("<|im_start|>user", "<|im_start|>assistant"),
        sep_style=SeparatorStyle.CHATML,
        sep="<|im_end|>",
        stop_str="<|im_end|>",
    )
)

# Deepseek-chat template
# reference: https://huggingface.co/deepseek-ai/deepseek-llm-67b-chat/blob/main/tokenizer_config.json
register_conv_template(
    Conversation(
        name="deepseek-chat",
        system_message="<｜begin▁of▁sentence｜>",  # must add a bos token before first message
        roles=("User", "Assistant"),
        sep_style=SeparatorStyle.DEEPSEEK_CHAT,
        sep="\n\n",
        sep2="<｜end▁of▁sentence｜>",
        stop_str="<｜end▁of▁sentence｜>",
    )
)

# Yuan2.0 chat template
# source: https://huggingface.co/IEITYuan/Yuan2-2B-Janus-hf/blob/main/tokenizer_config.json#L6
register_conv_template(
    Conversation(
        name="yuan2",
        roles=("user", "assistant"),
        sep_style=SeparatorStyle.YUAN2,
        sep="<sep>",
        sep2="\n",
        stop_token_ids=[
            77185,
        ],  # "<eod>"
        stop_str="<eod>",
    )
)

# Solar-10.7B Chat Template
# Reference: https://huggingface.co/upstage/SOLAR-10.7B-Instruct-v1.0/blob/main/tokenizer_config.json
register_conv_template(
    Conversation(
        name="solar",
        system_message="",
        roles=("### User", "### Assistant"),
        sep_style=SeparatorStyle.ADD_NEW_LINE_SINGLE,
        sep="\n\n",
        stop_str="</s>",
    )
)

# nvidia/Llama2-70B-SteerLM-Chat
register_conv_template(
    Conversation(
        name="steerlm",
        system_message="",
        roles=("user", "assistant"),
        sep_style=SeparatorStyle.DEFAULT,
        sep=None,
    )
)

# yuan 2.0 template
# reference:https://github.com/IEIT-Yuan/Yuan-2.0
# reference:https://huggingface.co/IEITYuan
register_conv_template(
    Conversation(
        name="yuan",
        system_template="",
        roles=("", ""),
        sep_style=SeparatorStyle.NO_COLON_SINGLE,
        sep="<sep>",
        stop_str="<eod>",
    )
)

# Cllm chat template
# reference:
register_conv_template(
    Conversation(
        name="cllm",
        system_message="A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions.",
        roles=("USER", "ASSISTANT"),
        sep_style=SeparatorStyle.CLLM,
        sep=" ",
        sep2="</s>",
    )
)


# Llava-chatml
# reference: https://github.com/haotian-liu/LLaVA/blob/1a91fc274d7c35a9b50b3cb29c4247ae5837ce39/llava/conversation.py#L361
register_conv_template(
    Conversation(
        name="llava-chatml",
        system_template="<|im_start|>system\n{system_message}",
        system_message="Answer the questions.",
        roles=("<|im_start|>user", "<|im_start|>assistant"),
        sep_style=SeparatorStyle.CHATML,
        sep="<|im_end|>",
        stop_str="<|im_end|>",
    )
)

# Gemma
# reference: https://huggingface.co/google/gemma-7b-it?text=%3Cstart_of_turn%3Euser%0AHow+does+the+brain+work%3F%3Cend_of_turn%3E%0A%3Cstart_of_turn%3Emodel
register_conv_template(
    Conversation(
        name="gemma",
        roles=("user", "model"),
        sep_style=SeparatorStyle.GEMMA,
        sep="<end_of_turn>\n",
        stop_str="<end_of_turn>",
    )
)

register_conv_template(
    Conversation(
        name="yandexgpt",
        system_message="",
        roles=("user", "assistant"),
        sep_style=None,
        sep=None,
    )
)

class ConversationForUI(Conversation):
    version: str = "Unknown"
    skip_next: bool = False

    def append_message(self, role, message):
        self.messages.append([role, message])

    def process_image(self, image, image_process_mode, return_pil=False, image_format='PNG', max_len=1344, min_len=672):
        if image_process_mode == "Pad":
            def expand2square(pil_img, background_color=(122, 116, 104)):
                width, height = pil_img.size
                if width == height:
                    return pil_img
                elif width > height:
                    result = Image.new(pil_img.mode, (width, width), background_color)
                    result.paste(pil_img, (0, (width - height) // 2))
                    return result
                else:
                    result = Image.new(pil_img.mode, (height, height), background_color)
                    result.paste(pil_img, ((height - width) // 2, 0))
                    return result
            image = expand2square(image)
        elif image_process_mode in ["Default", "Crop"]:
            pass
        elif image_process_mode == "Resize":
            image = image.resize((336, 336))
        else:
            raise ValueError(f"Invalid image_process_mode: {image_process_mode}")
        if max(image.size) > max_len:
            max_hw, min_hw = max(image.size), min(image.size)
            aspect_ratio = max_hw / min_hw
            shortest_edge = int(min(max_len / aspect_ratio, min_len, min_hw))
            longest_edge = int(shortest_edge * aspect_ratio)
            W, H = image.size
            if H > W:
                H, W = longest_edge, shortest_edge
            else:
                H, W = shortest_edge, longest_edge
            image = image.resize((W, H))
        if return_pil:
            return image
        else:
            buffered = BytesIO()
            image.save(buffered, format=image_format)
            img_b64_str = base64.b64encode(buffered.getvalue()).decode()
            return img_b64_str

    def get_images(self, return_pil=False):
        images = []
        for i, (role, msg) in enumerate(self.messages[self.offset:]):
            if i % 2 == 0:
                if type(msg) is tuple:
                    msg, image, image_process_mode = msg
                    image = self.process_image(image, image_process_mode, return_pil=return_pil)
                    images.append(image)
        return images

#TODO:adjust this fuction to video and audio.
    def to_gradio_chatbot(self):
        ret = []
        for i, (role, msg) in enumerate(self.messages[self.offset:]):
            if i % 2 == 0:
                if type(msg) is tuple:
                    msg, image, image_process_mode = msg
                    img_b64_str = self.process_image(
                        image, "Default", return_pil=False,
                        image_format='JPEG')
                    img_str = f'<img src="data:image/jpeg;base64,{img_b64_str}" alt="user upload image" />'
                    msg = img_str + msg.replace('<image>', '').strip()
                    ret.append([msg, None])
                else:
                    ret.append([msg, None])
            else:
                ret[-1][-1] = msg
        return ret

    def copy(self):
        return Conversation(
            system=self.system,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            version=self.version)

    def dict(self):
        if len(self.get_images()) > 0:
            return {
                "system": self.system,
                "roles": self.roles,
                "messages": [[x, y[0] if type(y) is tuple else y] for x, y in self.messages],
                "offset": self.offset,
                "sep": self.sep,
                "sep2": self.sep2,
            }
        return {
            "system": self.system,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
            "sep": self.sep,
            "sep2": self.sep2,
        }