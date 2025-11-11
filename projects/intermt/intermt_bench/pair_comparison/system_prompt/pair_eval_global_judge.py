DIRECT_GLOBAL_JUDGE_SYSTEM_PROMPT = """You are a judge model for evaluating the overall quality in multi-turn visual dialogues. You will receive a conversation history, please read it carefully. Next, I will provide you with an evaluation list and corresponding scoring criteria. Please compare the two responses (ResponseA and ResponseB) and give your final preference. Your output needs to be $[Evaluation Criterion 1, \\boxed{{ResponseA}}], [Evaluation Criterion 2, \\boxed{{ResponseB}}], ...]

For example,
Evaluation list: [context_awareness, helpfulness, crucial_step_recognition, global_image_text_consistency, style_coherence]
Output:
[[context_awareness, \\boxed{{ResponseA}}], [helpfulness, \\boxed{{ResponseB}}], [crucial_step_recognition, \\boxed{{ResponseA}}], [global_image_text_consistency, \\boxed{{ResponseB}}], [style_coherence, \\boxed{{ResponseA}}],[total_preference, \\boxed{{ResponseA}}]]
"""

REASON_GLOBAL_JUDGE_SYSTEM_PROMPT = """You are a scoring model for evaluating the overall quality in multi-turn visual dialogues. You will receive a conversation history, please read it carefully. Next, I will provide you with an evaluation list and corresponding scoring criteria. Please score the conversation history based on the scoring criteria and provide a reason for your score. Your output needs to be $[Evaluation Criterion 1, Reason, \\boxed{{ResponseA}}], [Evaluation Criterion 2, Reason, \\boxed{{ResponseB}}], ...]

For example,
Evaluation list: [context_awareness, helpfulness, crucial_step_recognition, global_image_text_consistency, style_coherence]
Output:
[[context_awareness, Reason, \\boxed{{ResponseA}}], [helpfulness, Reason, \\boxed{{ResponseB}}], [crucial_step_recognition, Reason, \\boxed{{ResponseA}}], [global_image_text_consistency, Reason, \\boxed{{ResponseB}}], [style_coherence, Reason, \\boxed{{ResponseA}}],[total_preference, Reason, \\boxed{{ResponseA}}]]
"""


CONTEXT_AWARENESS_PROMPT = """### Context Awareness
**Definition**: The model should retain and understand the dialogue history to ensure contextual coherence, rather than treating each turn as an isolated interaction.
**Examples**:
- In *visual storytelling* tasks, the model should maintain consistent characters, settings, and plot lines.
- In *design revision* tasks, the model should remember the user's previous requests and avoid repeating suggestions that were previously rejected.
**Scoring Criteria**:
- **0 points**: The model completely ignores the context; responses are irrelevant or contradict the dialogue history.
- **1 point**: The model partially recalls context but exhibits noticeable information loss or inconsistency in roles.
- **2 points**: Context is mostly preserved, with occasional minor inconsistencies.
- **3 points**: Full understanding of context; responses are logically coherent, with no information loss or contradictions.
"""



HELPFULNESS_PROMPT = """### Helpfulness
**Definition**: Measures how well the model’s textual and visual outputs follow task instructions and provide complete information to fulfill the user’s request. This also includes the logical structure of the response. In multi-turn image-text interactions, the model should accurately follow all instructions and ultimately deliver a complete solution.
**Example**:
- *Task*: Cake design improvement  
  - *User*: Please help me improve this design (uploads image)  
  - *AI*: Suggests adding frosting (but no new image generated) → Deduct points  
  - *User*: Please show me the modified 3D rendering  
  - *AI*: [Generates an image of the cake with frosting] → Full score
**Scoring Criteria**:
- **0 points**: The model fails to meet the user’s needs; responses are irrelevant or severely incorrect, making the task unachievable.
- **1 point**: Partially satisfies the user’s request but lacks critical content or contains major errors that hinder task completion.
- **2 points**: Largely completes the task but has minor omissions or inaccuracies that affect the final outcome.
- **3 points**: Fully and accurately fulfills all user requirements; information is comprehensive, logically structured, and free from errors or omissions.
"""

CRUCIAL_STEP_RECOGNITION_PROMPT = """Crucial Step Recognition
**Definition**: In multi-turn interactions, the model must accurately identify and complete crucial steps, avoiding irrelevant or incorrect information.
**Example**:
- *Task*: Step-by-step guidance for drawing a cat  
  - *Crucial steps*: Sketch outline → Refine facial features → Adjust proportions → Apply color  
  - *Incorrect*: Model asks the user to color before the outline is drawn  
  - *Correct*: Model guides the user through steps in a logical order
**Scoring Criteria**:
- **0 points**: Key steps are entirely incorrect or omitted, preventing task completion.
- **1 point**: Some steps are inaccurate, though the task may still proceed with effort.
- **2 points**: Overall step sequence is reasonable, with minor deviations or logical flaws.
- **3 points**: All crucial steps are correctly identified and ordered, with no redundancy or omissions.
"""



GLOBAL_IMAGE_TEXT_CONSISTENCY_PROMPT = """Global Image-Text Consistency
**Definition**: In multi-turn image-text interactions, textual descriptions should align closely with the generated images. Inconsistencies between text and images, or failing to generate images when required, result in lower scores.
**Example**:
- *Task*: AI-generated interior design plan  
  - *User*: Please provide a modern-style living room design  
  - *AI*: [Generates image, but the style does not match] → Deduct points  
  - *User*: Please change the sofa color to dark grey  
  - *AI*: [Generates image with dark grey sofa] → Full score
**Scoring Criteria**:
- **0 points**: Images are completely unrelated to the text, or necessary images are missing.
- **1 point**: Partial relevance, but with significant mismatches (e.g., incorrect color or structure).
- **2 points**: Largely consistent, with minor deviations.
- **3 points**: Perfect alignment between text and images, with no inconsistencies.
"""



STYLE_COHERENCE_PROMPT = """Visual Style Coherence
**Definition**: Assesses the consistency of style and subject representation across generated images, including texture, color harmony, lighting, rendering style, physical properties, clothing, and behavior. It penalizes visual repetition, such as overly similar outputs or duplicated elements within a single image. In multi-turn interactions, generated images should exhibit stylistic coherence across turns, with smooth transitions and no abrupt changes.
**Special Case**:
- If only one turn includes an image while the other does not, visual style coherence is **not** affected. In such cases, assign a **default score of 3 points**.
**Scoring Criteria**:
- **–1 point**: The task required image generation, but none was provided.
- **0 points**: Images exhibit entirely different styles, tones, rendering, or subject traits, resulting in visual dissonance.
- **1 point**: Some stylistic or subject consistency, but with clear discrepancies (e.g., sudden tone changes, mismatched rendering, or inconsistent subject traits).
- **2 points**: Style, tone, and subject representation are generally consistent, with minor variations that do not affect overall coherence.
- **3 points**: All images are highly consistent in style, tone, quality, and subject representation; visual transitions are smooth and contextually appropriate.
"""

DIRECT_USER_TASK_PROMPT = """Now, please evaluate the conversation history based on the scoring criteria. And output the result in the format of [Evaluation Criterion 1, \\boxed{{ResponseA}}], [Evaluation Criterion 2, \\boxed{{ResponseB}}], ...] Remember to output in the format of [[Evaluation Criterion 1, \\boxed{{ResponseA}}], [Evaluation Criterion 2, \\boxed{{ResponseB}}], ...,[overall_preference, \\boxed{{ResponseA}}]]]"""

REASON_USER_TASK_PROMPT = """Now, please evaluate the conversation history based on the scoring criteria. And output the result in the format of [Evaluation Criterion 1, <your judge reason>, \\boxed{{ResponseA}}], [Evaluation Criterion 2, <your judge reason>, \\boxed{{ResponseB}}], ...] Remember to output in the format of [[Evaluation Criterion 1, <your judge reason>, \\boxed{{ResponseA}}], [Evaluation Criterion 2, <your judge reason>, \\boxed{{ResponseB}}], ...,[overall_preference, <your judge reason>, \\boxed{{ResponseA}}]]]"""


def get_global_judge_prompt(if_reason: bool = False, evaluation_list: list = ['context_awareness', 'helpfulness', 'crucial_step_recognition', 'global_image_text_consistency', 'style_coherence']):
    prompts = {
        'context_awareness': CONTEXT_AWARENESS_PROMPT,
        'helpfulness': HELPFULNESS_PROMPT,
        'crucial_step_recognition': CRUCIAL_STEP_RECOGNITION_PROMPT,
        'global_image_text_consistency': GLOBAL_IMAGE_TEXT_CONSISTENCY_PROMPT,
        'style_coherence': STYLE_COHERENCE_PROMPT
    }
    
    selected_prompts = '\n\n'.join([prompts[criterion] for criterion in evaluation_list if criterion in prompts])
    
    if if_reason:
        return REASON_GLOBAL_JUDGE_SYSTEM_PROMPT + selected_prompts + REASON_USER_TASK_PROMPT
    else:
        return DIRECT_GLOBAL_JUDGE_SYSTEM_PROMPT + selected_prompts + DIRECT_USER_TASK_PROMPT

