import os
import sys
import json
from tqdm import tqdm
from datasets import load_dataset


def inference_with_agent(question):
    """
    Implement the inference process of the all-modality generation model.
    """
    
    template_tia = """
    You are a helpful AI. You are now going to answer the user's question while generating two sets of instructions: one for a image generation model and one for an audio generation model. Make sure:
    1.	Your text response should be clear and precise, referencing the generated image and audio content as much as possible in the answer.
    2.	The instructions for the image generation should be accurate and concise.
    3.	The instructions for the audio generation should be accurate and concise, no more than 20 words.

    Your output should follow the following format:
    [output_format]
    [[response]]: <response>
    [[image_instruction]]: <image_instruction>
    [[audio_instruction]]: <audio_instruction>
    [/output_format]

    [User question]
    {question}
    [/User question]
    """
    
    prompt = template_tia.format(question = question)
    pass

def inference(prompt):
    """
    Implement the inference process of the all-modality generation model.
    If you have created the truly all-modality generation model and use this part for evaluation,
    I would like to thank you very much for your outstanding contributions to the community's development.
    """
    pass

def main():
    dataset = load_dataset(
        "PKU-Alignment/EvalAnything-Selection_Synergy", 
        trust_remote_code=True
    )

    save_dir = os.path.join("amg", "synergy")
    os.makedirs(save_dir, exist_ok=True)

    results = []
    for entry in tqdm(dataset):
        instruction = entry["instruction"]
        
        result_entry = inference_with_agent(
            instruction
        )
        
        results.append({
            "prompt": instruction,
            "output": result_entry
        }) 
        
    with open(os.path.join(save_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=4)


if __name__ == '__main__':
    main()

