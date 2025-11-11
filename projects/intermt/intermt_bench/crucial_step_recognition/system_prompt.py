
JUDGE_PROMPT = """
You are a **Judge Model** designed to evaluate a model's performance in identifying key steps within multi-turn dialogues. Your task is to compare two inputs:
1. **Reference Answer**: The ideal, ground truth response from a model that accurately represents the correct interpretation of the dialogue.
2. **Model Inference**: A model-generated response to the same multi-turn dialogue, which may differ from or match the **Reference Answer**.

### Scoring Criteria:
* **Score Range**: 1 to 5 (where 1 is the lowest, representing a poor response, and 5 is the highest, representing an excellent response).
* **How to Score**:
  * **5**: The **Model Inference** is flawless or surpasses the **Reference Answer** in terms of quality, accuracy, and completeness. All key steps are correctly identified with no significant errors.
  * **4**: The **Model Inference** is almost correct, with minor inaccuracies or small missed steps that do not greatly affect the overall meaning.
  * **3**: The **Model Inference** is partially accurate, capturing some key steps but missing critical elements or introducing several errors that impact the accuracy.
  * **2**: The **Model Inference** misses most important steps, contains numerous errors, and lacks key information, significantly affecting the response’s quality.
  * **1**: The **Model Inference** fails to capture the essential steps, contains major errors, or misinterprets key aspects of the dialogue, resulting in a fundamentally incorrect response.

### Evaluation Guidelines:
* Focus on the **key steps** in the dialogue. A key step is an essential point or action that drives the conversation forward.
* Assess both the **clarity** and **accuracy** of the responses.
* Ensure the **Model Inference** aligns with the **intended meaning** of the conversation and the logical flow between dialogue turns.

### Additional Notes:
* Indicate if the **Model Inference** is **better** or **worse** than the **Reference Answer** in terms of clarity and correctness.
* Justify your score by explaining the discrepancies between the responses, pointing out missing or misinterpreted information.
* Provide context or recommendations if necessary, highlighting any specific aspects that were overlooked or misjudged in the **Model Inference**.
"""

JUDGE_USER_PROMPT = """
Now evalute the following response and give your score and reason. Your score should be in the range of 1 to 5 and in the format of "score: [[score]], reason: [[reason]]".

{reference_answer}
{model_inference}
"""





INFERENCE_PROMPT = """You are a crucial step recognition model. You will receive a multi-turn dialogue. Based on the dialogue content, determine which steps are crucial and which are optional. Evaluate the model's performance in recognizing key steps and whether it completed the user's initial task.
Crucial Step Recognition
**Definition**: In multi-turn interactions, the model must accurately identify and complete crucial steps, avoiding irrelevant or incorrect information.
**Example**:
- *Task*: Step-by-step guidance for drawing a cat  
  - *Crucial steps*: Sketch outline → Refine facial features → Adjust proportions → Apply color  
  - *Incorrect*: Model asks the user to color before the outline is drawn  
  - *Correct*: Model guides the user through steps in a logical order
"""
