import os
import json
import torch
import scipy

from tqdm import tqdm
from datasets import load_dataset

from diffusers import AudioLDM2Pipeline


repo_id = "cvssp/audioldm2"
pipe = AudioLDM2Pipeline.from_pretrained(repo_id, torch_dtype=torch.float16)

USER_PROMPT_TEMPLATE = """\
You are about to receive a user instruction. Please convert the user instruction into an instruction to an audio generation model. Make sure that the instruction for the audio generation should be accurate and concise, no more than 40 words. You should only output the instruction prompt.

[User Instruction]
{prompt}
[/User Instruction]
"""

def generate_audio_instruction(prompt):
    """
    Implement the inference process of the audio generation part of the model.
    """
    pass

def inference(data):
    """
    Implement the inference process of the audio generation part of the model.
    """
    prompt = USER_PROMPT_TEMPLATE.format(prompt=data['prompt'])
    audio_instruction = generate_audio_instruction(prompt)
    
    audio = pipe(audio_instruction, num_inference_steps=200, audio_length_in_s=10.0).audios[0]
    return audio


dataset = load_dataset(
    "PKU-Alignment/EvalAnything-InstructionFollowing",
    name="audio_instruct",
    split="test",
    trust_remote_code=True
)


os.makedirs(".cache/audio_instruct", exist_ok=True)

results = {}

for data in tqdm(dataset):
    audio = inference(data)
    scipy.io.wavfile.write(f".cache/audio_instruct/{data['prompt_id']}.wav", rate=16000, data=audio)
    
    results[data['prompt_id']] = f".cache/audio_instruct/{data['prompt_id']}.wav"

with open(".cache/audio_instruct/generated_results.json", "w") as f:
    json.dump(results, f)
