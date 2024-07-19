from outputs import Arena_input, InferenceOutput
from eval.base_eval_api import GPT_eval
Inferenced = [
    InferenceOutput.from_dict(
        data = {
            'prompt':"Hi?",
            'response':"Hello!",
        }
    ),
    InferenceOutput.from_dict(
        data = {
            'prompt':"Hi?",
            'response':"good!"
        }
    )
]
dats = Arena_input.from_InferenceOutput(inference1=Inferenced[0], inference2=Inferenced[1])
print(dats)
evaluator = GPT_eval(
    system_prompt="Choose which response is better, response me with the better one, reponse 'A' or 'B'",
    api_key="sk-84f6b26633ed46f7bccb7afec95cd06c",
    base_url="https://api.deepseek.com",
    cache_dir="./cache",
    model="deepseek-chat",
    num_workers=1
)
print(isinstance(dats, Arena_input))
evaluator_output = evaluator.evaluate(inputs=dats)
print(evaluator_output)