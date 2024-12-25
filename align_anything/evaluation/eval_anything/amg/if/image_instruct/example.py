import os
import json
import torch
import scipy

from tqdm import tqdm
from datasets import load_dataset

import torch
from diffusers import FluxPipeline

pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload() 

USER_PROMPT_TEMPLATE = """\
USER:
You are about to receive a user instruction. Please convert the user instruction into an instruction to an image generation model. Make sure that the instruction for the image generation should be accurate and concise, no more than 40 words. You should only output the instruction prompt.

[User Instruction]
{prompt}
[/User Instruction]
ASSISTANT:
"""

def generate_image_instruction(prompt):
    """
    Implement the inference process of the image generation part of the model.
    """
    pass

def inference(data):
    """
    Implement the inference process of the audio generation part of the model.
    """
    prompt = USER_PROMPT_TEMPLATE.format(prompt=data['prompt'])
    image_instruction = generate_image_instruction(prompt)
    
    image = pipe(
        image_instruction,
        guidance_scale=0.0,
        num_inference_steps=4,
        max_sequence_length=256,
        generator=torch.Generator("cpu").manual_seed(0)
    ).images[0]
    return image


dataset = load_dataset(
    "PKU-Alignment/EvalAnything-InstructionFollowing",
    name="image_instruct",
    split="test",
    trust_remote_code=True
)


os.makedirs(".cache/image_instruct", exist_ok=True)

results = []

for data in tqdm(dataset):
    image = inference(data)
    image.save(f".cache/image_instruct/{data['prompt_id']}.png")
    
    results.append({
        "prompt_id": data['prompt_id'],
        "prompt": data['prompt'],
        "image_path": f".cache/image_instruct/{data['prompt_id']}.png"
    })

with open(".cache/image_instruct/generated_results.json", "w") as f:
    json.dump(results, f)
