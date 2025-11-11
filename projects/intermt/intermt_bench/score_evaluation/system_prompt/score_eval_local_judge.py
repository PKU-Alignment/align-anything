# Evaluate each turn data separately, but each data evaluates multiple metrics simultaneously, 
# then output average agreement for each metric of each conversation recorded in debug.json, 
# and output overall average agreement for all metrics across all turns recorded in clean.json and accuracy.json

DIRECT_LOCAL_JUDGE_SYSTEM_PROMPT = """You are a scoring model for evaluating the quality of a single turn in multi-turn visual dialogues. You will receive a conversation history, please read it carefully. Next, I will provide you with an evaluation list and corresponding scoring criteria. Please score the conversation history based on the scoring criteria. Your output must follow this exact format:

Evaluation list: [local_image_text_consistency, perceptual_quality, text_quality, contextual_coherence]

[[local_image_text_consistency, reason, \\boxed{score}]]
[[visual_perceptual_quality, reason, \\boxed{score}]]
[[text_quality, reason, \\boxed{score}]]
[[context_coherence, reason, \\boxed{score}]]

Where "score" is your numerical rating (0-3) and "reason" is your brief justification.
"""



LOCAL_IMAGE_TEXT_CONSISTENCY_PROMPT = """Local Image-Text Consistency
**Definition**: In a single dialogue turn, the textual description should closely match the generated image(s), ensuring the text accurately reflects the visual content without ambiguity or misleading information.
**Applicable Scenarios**:
- If the turn includes multiple images, evaluate the overall consistency of the text with all images. Individual image feedback can be added as needed (e.g., `[3,1] Image 1: accurate; Image 2: inconsistent`).
- If no image is generated, evaluate based on task requirements:
  - If image generation was expected but omitted, assess the inconsistency between the text and the missing visual content.
  - If the task (e.g., Visual Analysis) does not require image generation, assess consistency between the input image and the text.
- Otherwise, default evaluation compares the answer text with the image(s) generated in that turn.
**Scoring Criteria**:
- **0 points**:
  - Text is irrelevant to the image(s) or contains major factual errors;
  - Key descriptions are missing or completely incorrect (e.g., referencing nonexistent objects or scenes);
  - Text may cause significant misunderstanding.
- **1 point**:
  - Text is partially related to the image(s), but includes clear errors or misleading descriptions;
  - Covers part of the image content but omits or misrepresents key details or relationships;
  - Reader must infer or adjust understanding to align with the image(s).
- **2 points**:
  - Text generally matches the image(s), with minor local inaccuracies (e.g., imprecise attribute descriptions or slight omissions);
  - Does not hinder overall comprehension, but lacks precision upon close inspection.
- **3 points**:
  - Text is highly aligned with the image(s), covering all key elements and details;
  - Free from factual errors or ambiguity; the description is natural and coherent.
"""


VISUAL_PERCEPTUAL_QUALITY_PROMPT = """Visual Perceptual Quality
**Definition**: Evaluates the visual realism, naturalness, and absence of distortion or artifacts in the generated image(s). Focuses on whether the image structure, colors, and composition realistically simulate the physical world, avoiding unnatural artifacts.
**Applicable Scenarios**:
- In multi-image outputs, assign a unified score for overall quality. If image quality varies significantly, provide per-image feedback as needed (e.g., `[3,1] Image 1: good; Image 2: distorted`).
- If no image is generated, assess any image provided in the user prompt. If the image has issues, point them out in the textual answer.
**Scoring Criteria**:
- **0 points**:
  - Obvious artifacts (e.g., disconnections, misalignments), severe distortions (e.g., highly unrealistic shapes), or structural errors (e.g., unbalanced proportions, illogical composition);
  - Unnatural color rendering (e.g., harsh color blocks, abnormal tones);
  - Lighting does not follow physical laws, severely affecting image recognizability.
- **1 point**:
  - Image is mostly recognizable but contains localized severe flaws;
  - Examples: anatomical errors (e.g., limb dislocation), inconsistent local color (e.g., banding, strong noise), or small rendering failures;
  - Overall naturalness is compromised, affecting visual coherence.
- **2 points**:
  - Image is generally natural and coherent; structure, color, and lighting are mostly reasonable;
  - Minor local imperfections such as rough edges, small artifacts, or slight blurring that do not affect overall perceptual quality.
- **3 points**:
  - Image is visually realistic and natural;
  - Well-structured, smooth color transitions, physically consistent lighting;
  - No visible artifacts, distortions, or flaws; overall aesthetics and details are excellent.
"""




TEXT_QUALITY_PROMPT = """Text Quality
**Definition**: Measures the clarity, coherence, and correctness of the output text. Includes grammar, spelling, readability, consistency with instructions and context, and absence of redundancy. Responses should be logically sound, well-structured, and clearly expressed, avoiding abrupt transitions or repetition.
**Scoring Criteria**:
- **0 points**: Text is disorganized, lacks logic, and is hard to understand; may contain numerous grammar or spelling errors or repetitive content.
- **1 point**: Some parts are logically clear, but the text includes noticeable jumps, omissions, or contradictions that hurt overall readability; may include frequent language errors or redundant expressions.
- **2 points**: The overall logic is reasonable and the flow mostly smooth, but there are minor incoherences; some sentences may require optimization to improve readability.
- **3 points**: Text is logically rigorous, clearly expressed, well-organized, and naturally structured; no obvious jumps or repetition; grammar and spelling are correct, providing a good reading experience.
"""


CONTEXT_COHERENCE_PROMPT = """Context Coherence
**Definition**: Assesses whether the response in this turn logically continues the dialogue history and remains consistent with prior content, avoiding contradictions.
**Scoring Criteria**:
- **0 points**: Completely irrelevant or logically inconsistent with previous context.
- **1 point**: Partially relevant but includes clear inconsistencies.
- **2 points**: Mostly coherent, with minor deviations.
- **3 points**: Fully consistent with prior dialogue; no contradictions.
"""

DIRECT_USER_TASK_PROMPT = """Now, please evaluate the this turn based on the scoring criteria. Your score should be between 0 and 3. And output the result in the format of [Evaluation Criterion 1, \\boxed{{score1}}], [Evaluation Criterion 2, \\boxed{{score2}}], ...]"""

REASON_USER_TASK_PROMPT = """Now, please evaluate the this turn based on the scoring criteria. Your score should be between 0 and 3. And output the result in the format of [Evaluation Criterion 1, Reason, \\boxed{{score1}}], [Evaluation Criterion 2, Reason, \\boxed{{score2}}], ...]"""


def get_local_judge_prompt(if_reason: bool = False, evaluation_list: list = ['local_image_text_consistency', 'visual_perceptual_quality', 'text_quality', 'context_coherence']):
    prompts = {
        'local_image_text_consistency': LOCAL_IMAGE_TEXT_CONSISTENCY_PROMPT,
        'visual_perceptual_quality': VISUAL_PERCEPTUAL_QUALITY_PROMPT,
        'text_quality': TEXT_QUALITY_PROMPT,
        'context_coherence': CONTEXT_COHERENCE_PROMPT
    }
    selected_prompts = '\n\n'.join([prompts[criterion] for criterion in evaluation_list if criterion in prompts])
    
    if if_reason:
        return DIRECT_LOCAL_JUDGE_SYSTEM_PROMPT + selected_prompts + REASON_USER_TASK_PROMPT
    else:
        return DIRECT_LOCAL_JUDGE_SYSTEM_PROMPT + selected_prompts + DIRECT_USER_TASK_PROMPT
    











