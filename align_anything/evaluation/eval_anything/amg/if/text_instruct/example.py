import json
import os

from tqdm import tqdm

from datasets import load_dataset


def generate_text_instruction(prompt):
    """
    Implement the inference process of the text generation part of the model.
    """


def inference(data):
    """
    Implement the inference process of the text generation part of the model.
    """
    return generate_text_instruction(data['prompt'])


dataset = load_dataset(
    'PKU-Alignment/EvalAnything-InstructionFollowing',
    name='text_instruct',
    split='test',
    trust_remote_code=True,
)


os.makedirs('.cache/text_instruct', exist_ok=True)

results = []

for data in tqdm(dataset):
    response = inference(data)
    results.append({'prompt_id': data['prompt_id'], 'prompt': data['prompt'], 'response': response})

with open('.cache/text_instruct/generated_results.json', 'w') as f:
    json.dump(results, f)
