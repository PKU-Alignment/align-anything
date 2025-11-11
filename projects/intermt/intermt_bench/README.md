*Do MLLMs truly understand what is desirable in multi-turn, multimodal interactions and how to align with human values?* This task is particularly challenging due to the absence of multimodal benchmarks that capture human preferences in multi-turn settings. Inspired by LLM-as-a-Judge and MLLM-as-a-Judge and leveraging genuine feedback from **InterMT**, we introduce **InterMT-Bench** to assess MLLMs' alignment with human values in multi-turn, multimodal tasks.

InterMT-Bench comprises three distinct tasks: *Scoring Evaluation*, *Pair Comparison*, and *Crucial Step Recognition*.

The test dataset includes multi-turn multimodal interleaved communication histories and human-annotated ground truth. Evaluated models must assess the conversation at both the turn and conversation levels across nine dimensions, following a set of guidelines. *Scoring Evaluation* requires the model to assign scores on a 0-3 scale, with evaluation based on agreement and Pearson similarity. *Pair Comparison* directly compares two individual turns or entire conversations, without considering ties, and is evaluated for accuracy against human judgments. *Crucial Step Recognition* addresses a key challenge in multi-turn conversations: accurately identifying the user's intent and determining whether it has been fulfilled, evaluated by the score provided by judge according to the human-annotated reference answers.


Image Path: https://huggingface.co/datasets/PKU-Alignment/InterMT-Bench-Images
