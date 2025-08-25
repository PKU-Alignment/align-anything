# Technical Report

In July 2024, we launched Align-Anything, an open-source initiative aimed at aligning any-to-any modality models with human values. It serves as a foundational infrastructure, providing the community with training code, datasets, models, and evaluation code support.

Today, we are excited to present the technical report of Align-Anything, aiming to concisely share our experiences gained in research and engineering for all-modality alignment.

Align-Anything aims to align any modality large models (any-to-any models) with human intentions and values. 

- **Highly Modular Framework** allowing users to easily modify and customize the code for different tasks (see [framework design](https://align-anything.readthedocs.io/)).
- **Various Modality Model Fine-Tuning** for diverse multi-modal (image/video/audio) models (see [scripts]([./scripts](https://github.com/PKU-Alignment/align-anything/tree/main/scripts))).
- **Different Alignment Methods.** Different alignment algorithms, including SFT, DPO, PPO, and others.
- **Rule-based RL.** Rule-based RL encouraged by [Deepseek-R1](https://huggingface.co/deepseek-ai/DeepSeek-R1).
- **Any-to-Any Modality Fine-tuning.** Tracking the performance of all modality large models at [Eval-Anything](https://github.com/PKU-Alignment/eval-anything).

Below is the detailed technical report

- [Technical Report](#technical-report)
  - [Open-Sourced Framework](#open-sourced-framework)
    - [Training Part](#training-part)
    - [Evaluation Part](#evaluation-part)
  - [Algorithm: Learning from Language Feedback (LLF)](#algorithm-learning-from-language-feedback-llf)
    - [Methods](#methods)
    - [Experiment](#experiment)
  - [Benchmark: Any-to-Any Modality Understanding and Generation](#benchmark-any-to-any-modality-understanding-and-generation)
    - [All-Modality Understanding](#all-modality-understanding)
    - [All-Modality Generation](#all-modality-generation)
    - [Results and Analysis](#results-and-analysis)
      - [Human and AI Agreement](#human-and-ai-agreement)
      - [Input vs Outputs](#input-vs-outputs)
      - [Truly All-Modality Model](#truly-all-modality-model)
  - [Dataset: Align-Anything 200K](#dataset-align-anything-200k)
    - [Current Open-Source Datasets](#current-open-source-datasets)
    - [Highlights](#highlights)
      - [All-Modality Tasks](#all-modality-tasks)
      - [Fine-Grained Preference](#fine-grained-preference)
      - [Language Feedback](#language-feedback)
    - [Annotation Pipeline](#annotation-pipeline)
      - [Collect Q-A Pairs](#collect-q-a-pairs)
      - [Fine-grained Annotation](#fine-grained-annotation)
      - [Language Feedback Generation](#language-feedback-generation)
    - [Datasets Comparison](#datasets-comparison)
    - [Human Agreement Analysis](#human-agreement-analysis)
  - [Enhancing Agent Reasoning with Verifiable Rewards and RAGEN](#enhancing-agent-reasoning-with-verifiable-rewards-and-ragen)
    - [Key Features](#key-features)
    - [Experimental Results](#experimental-results)


## Open-Sourced Framework

### Training Part

Align-Anything aims to align any modality large models (any-to-any models), including LLMs, VLMs, and others, with human intentions and values. The align-anything framework integrates all-modality alignment algorithms (*e.g.,* Reinforcement Learning from Human Feedback, RLHF), supporting SFT, RM, DPO, and PPO. Additionally, align-anything implements KTO, SimPO, and ORPO in the text-to-text modality. Besides, align-anything offers a highly scalable model registration mechanism and currently supports the training and deploying over 25 models. 

![1](https://align-anything.readthedocs.io/en/latest/_images/framework.png)

What align-anything does is to use a framework to scale up the simple process above to any modality model. The underlying logic of this framework is precisely the four steps mentioned above. We will provide the align-anything implementation of LLaVA’s SFT code here, and implementations for other modalities are similar. After reading this example, you should fully understand the usage and underlying logic of align-anything.

![2](https://align-anything.readthedocs.io/en/latest/_images/example.png)


### Evaluation Part

[**Eval-anything**](https://github.com/PKU-Alignment/eval-anything) aims to track the performance of all modality large models (any-to-any models) on safety tasks and evaluate their true capabilities.

* **Datasets**

  - **Self-developed Dataset**: A dataset specifically designed for assessing all-modality safety of large models.

  - **Integration of Over 50 Open-source Datasets**: Diverse data sources for comprehensive safety assessment.

  - **Five Core Evaluation Dimensions** with 35 sub-dimensions.

- **Embodied Safety Evaluation Framework:**
  - **Covering Various Modality Evaluations**: Text, image, video, speech, and action.
  - **Defining Major Task Categories in Embodied Safety**: Corner cases, blind spots, fragile collections, critical points, and dangerous equipment.
  - **Proposing Major Goals of Embodied Safety Evaluation**: Execution safety, long-range trajectory safety, and hardware safety.

* **Platform Integration**
  - Eval-anything seamlessly integrates with [FlagEval](https://flageval.baai.ac.cn/) to enhance assessment effectiveness.

## Algorithm: Learning from Language Feedback (LLF)

### Methods

![3](https://github.com/Gaiejj/align-anything-images/blob/main/docs/algo.png?raw=true)

Inspired by [Constitutional AI](https://www.anthropic.com/research/constitutional-ai-harmlessness-from-ai-feedback), LLF comprises two steps: *feedback modeling* and *self-improving*. The former employs SFT to train the feedback model, enabling it to provide language feedback based on $x$ and $y$. The latter allows the model to refine its responses based on the language feedback $c$.

**Feedback Modeling** The training process utilizes a dataset 

$$D = (x_i, y_i, c_i)_{i=1}^{N}$$

where $N$ is the size of dataset, $x_i$ denotes the prompt, $y_i$ represents the response, and $c_i$ is the corresponding language feedback. Let $P_\varphi(c_i | x_i, y_i)$ denote the probability of the target sequence $c_i$ given the input sequence $(x_i, y_i)$ and the model parameters $\varphi$, the training objective of the feedback model can be expressed by the cross-entropy loss:

$$L_{\varphi} = - E_{(x_i, y_i, c_i) \sim D}[ log P_{\varphi}(c_i | x_i, y_i)]$$

**Self Improving** We first collect the initial model's response $y$ to the given prompt $x$ online. Then, we gather feedback $c=M_{\phi}(x, y)$ from the feedback model $M_{\phi}$. Finally, we have the initial model generate responses 

$$y^{*}=M_{\theta}(x,c)$$

conditioned on this feedback. For example, $y$ may suffer from redundancy and hallucination, while $c$ will remind the generation to avoid these problems. Based on online sampling, it often enables two responses to exhibit significant differences in certain aspects (*e.g.,* reducing redundancy), thereby generating more learnable preference pairs.

### Experiment

![4](https://github.com/Gaiejj/align-anything-images/blob/main/docs/llf.png?raw=true)

We empirically verify that LLF offers several key advantages over traditional binary feedback: *unified preference*: As language feedback typically optimizes responses along key dimensions, models can easily learn genuine human preference beyond pairs; *rich information*: Since the feedback model can generate a substantial amount of language feedback, LLF can effectively serve as a robust synthesizer of preference data. Specifically, we train Chameleon-7B for T2I and TI2TI, LLaVA-7B and LLaVA-13B for TI2T, Qwen2-Audio-7B for TA2T, and Qwen2-VL-7B for TV2T based on *align-anything-200k*.

**Improvement with Unified Preference** LLF synthesized preference pairs reflect more unified human preference, enhancing all-modality alignment performance. We observe that DPO and RLHF using binary pairs fall short in some modalities. However, with LLF, they yields positive improvement across all modalities. Interestingly, we find that the improvement of LLF on LLaVA-13B is greater than that on LLaVA-7B. This suggests that LLF performs better on stronger models.

![5](https://github.com/Gaiejj/align-anything-images/blob/main/docs/exp_2.png?raw=true)

**Efficiency with Rich Information** LLF provides richer information and supports efficient preference data synthesis. Despite having only a smaller amount of language feedback, DPO+LLF outperforms DPO with binary feedback. At the same time, the reduction in data amount does not significantly weaken the capabilities of LLF. This suggests that in all-modality alignment, labeling many binary preference pairs is less effective than labeling a smaller amount of language feedback.

## Benchmark: Any-to-Any Modality Understanding and Generation

Currently, evaluating all-modality models relies on human experts for assessments, which is inefficient and costly. While combining benchmarks for individual modalities could offer a broader evaluation, differences in data preparation, post-processing, and metrics across benchmarks hinder accurate performance assessment. Additionally, all-modality models uniquely select the appropriate modalities based on user queries, enabling seamless cross-modal synergy, a capability that traditional single-modality evaluation pipelines fail to capture fully.

To address this gap, we deliver our evaluation framework specifically designed for all-modality models -- *eval-anything* -- including (1) **all-modality understanding** (AMU) for assess models to simultaneously process and integrate information from all modalities and (2) **all-modality generation** (AMG): evaluate a model's ability to follow user instructions, autonomously select modalities, and work synergistically across different modalities for output.

![6](https://github.com/Gaiejj/align-anything-images/blob/main/docs/eval.png?raw=true)

### All-Modality Understanding

All-modality models aim to both understand individual modalities and combine information across them to generate high-quality responses. To assess their comprehensive multimodal processing, we create 164 test entries, each containing textual, visual (image or video), and auditory (audio or speech) components. These interconnected modalities require the model to integrate all inputs accurately, as failure in any one modality leads to incorrect answers. If a model fails to process visual inputs, it may miss that *the person in the picture is frightened*. Similarly, without auditory processing, it might not understand that the person's fear is due to *a barking dog*.

We use GPT-4 to evaluate model responses on a scale of 1 to 10. However, previous studies underscore limitations in multimodal evaluation that rely on a single human annotation as a reference. As the number of modalities increases, so does the complexity, making it harder to reach consensus and increasing subjective bias in single annotations. Moreover, since GPT-4 lacks true multimodal comprehension, evaluating only its text responses doesn't confirm if essential information from each modality is fully understood. To address this, we gather responses from 10 annotators and extract key terms from each modality. The distribution of multiple annotations mitigates the bias associated with single-reference evaluation. Additionally, the inclusion of key terms enables GPT-4 to more accurately detect potential errors in its responses.

### All-Modality Generation

In generation tasks, all-modality models can outperform single-modal ones by delivering diverse information across multiple formats under the same user query. To achieve this, they must follow ***SSI*** principles: (1) **S**elect relevant modalities automatically to reduce redundancy, (2) **S**ynergistic integration for maximum information gain, and (3) **I**nstruction-following in each modality. The overall score for the AMG task is as follows:

$$S_\text{AMG} = \sum_{T, V, A} \frac{1}{2} \alpha_{T, V} \cdot \left( S_T + S_V \right) \cdot S_{T, V}$$

**Instruction Following** To assess instruction-following, we design 100 prompts per modality. Using GPT-4o, we score the text and image outputs. For audio and video modalities, inspired by TIFA pipeline, we create multiple-choice questions based on the prompts and employ Qwen2-Audio and Qwen2-VL as evaluation models. This process yields independent modality scores $S_T, S_V, S_A$, each ranging from 0 to 10.

**Modality Selection** Properly combining modalities in the model's response can provide rich perspectives while reducing redundant information. We employ 25 crowdsourced annotators to identify the expected modality combinations for given text instructions. The voting process results in one of the quantitative metrics for AMG task, with appropriate rewards and penalties applied based on the modalities generated by the model. The $\alpha$ represents the percentage of human votes for each modality combination, and if the model outputs all three modalities, this parameter will be one-third of $\alpha_{T, V, A}$.

**Modality Synergy** refers to the consistency between different modalities in a model's responses. We assess modality synergy by training a judge model with human annotations obtained through a data synthesis. Specifically, we develop a data synthesis pipeline centered on LLM-based agents that utilize tools to invoke audio, image, and video generation models, constructing a dataset with textual inputs and multimodal outputs. We employ human annotators to annotate preference data considering the synergy between different modalities in each response, and then train a judge model that allows all-modality input. Responses with high synergy scores ($S_{T, V}, S_{T, A}, S_{V, A}$) should show high relevance and consistency between modalities, with visual and auditory elements enriching the text from various perspectives.

### Results and Analysis

![7](https://github.com/Gaiejj/align-anything-images/blob/main/docs/eval_exp.png?raw=true)

#### Human and AI Agreement

In the modality synergy task, after training on the 5k preference dataset, the experiment reveals a 66.4% agreement rate between the judging model and human annotators. These figures are consistent with human agreement ratios reported in similar studies on modeling human preferences in the multimodal large language models domain.

#### Input vs Outputs

Most models in the main evaluation results support partial modality input and have baseline scores, but Gemini-1.5-Pro outperforms others due to its ability to process all three modalities.
In the AMG task, the average scores is relatively low, with no model demonstrating a clear advantage across all sub-items.
The results indicate that, compared to modality generation, models are more advanced in modality understanding, consistent with the developmental trends of all-modality models.

#### Truly All-Modality Model

Current models still fall far behind all-modality models. For all-modality understanding, models using only a single modality score less than half the maximum points, as shown in the main results table. Even Gemini-1.5-Pro, which processes both visual and auditory inputs, fails to attain a perfect score. Unlike humans, who integrate information from multiple modalities, these models are limited by their inability to perceive different modalities in a fully integrated way, even though they perform nearly at the human level within individual modalities. Moreover, models align poorly with human choices when selecting output modalities, as illustrated in the modality selection comparison figure. Humans can adaptively choose the best combination of modalities based on the instruction. In contrast, models tend to output either only textual information, resulting in information loss, or all available modalities, causing redundancy. The limited multimodal capability, especially in models trained mainly on text, hampers their ability to synthesize information effectively, making it difficult to balance detail and conciseness.

## Dataset: Align-Anything 200K

Our world is inherently multimodal. Humans perceive the world through multiple senses, and **Language Models** should operate similarly. However, the development of **Current Multi-Modality Foundation Models** faces limitations due to the availability and diversity of data across different modalities. Specifically, the challenges include:

1. **Imbalance in modality data**: While there is abundant data for vision tasks, data for other modalities such as video and audio is relatively scarce, and there is a lack of interconnected data across different modalities.
2. **Limited multi-modality training data**: The majority of existing datasets focus on modality-specific question-answer tasks, while there is a lack of specialized datasets to enhance multi-modality models' **Instruction-Following** capabilities.

To address these challenges, we propose **Align-Anything 200K**, which features:
- **All-modality tasks**: Incorporating tasks that cover all major modalities.
- **Fine-grained preference**: Capturing nuanced user preferences across tasks.
- **Language feedback**: Supporting critique and refinement through natural language.
- **Cross-modality QA pairs**: Enabling richer interactions between different modalities.

### Current Open-Source Datasets

You can click the links in `Modality Type` for more details.
| Modality Type | Dataset Type         | Current Open-source Data Volume |
|---------------|----------------------|---------------------------------|
| [Text-to-Text](https://huggingface.co/datasets/PKU-Alignment/align-anything/blob/main/text-to-text/README.md)           | Preference           | 30K                             |
| [Text-Image-to-Text](https://huggingface.co/datasets/PKU-Alignment/align-anything/blob/main/text-image-to-text/README.md)          | Preference           | 40K                             |
| [Text-Image-to-Text-Image](https://huggingface.co/datasets/PKU-Alignment/align-anything/blob/main/text-image-to-text-image/README.md)         | Preference           | 27K                             |
| [Text-to-Image](https://huggingface.co/datasets/PKU-Alignment/align-anything/blob/main/text-to-image/README.md)           | Preference           | 32K                             |
| [Text-Audio-to-Text](https://huggingface.co/datasets/PKU-Alignment/align-anything/blob/main/text-audio-to-text/README.md)         | Preference           | 30K                             |
| [Text-to-Audio](https://huggingface.co/datasets/PKU-Alignment/align-anything/blob/main/text-to-audio/README.md)           | Preference           | 12K                             |
| [Text-to-Video](https://huggingface.co/datasets/PKU-Alignment/align-anything/blob/main/text-to-video/README.md)           | Preference           | 9K                              |
| [Text-Video-to-Text](https://huggingface.co/datasets/PKU-Alignment/align-anything/blob/main/text-video-to-text/README.md)          | Preference           | 10K                             |
| [Text-Image-to-Text-Instruction](https://huggingface.co/datasets/PKU-Alignment/Align-Anything-TI2T-Instruction-100K)          | Instruction-Following | 100K                            |
| [Text-to-Text-Instruction](https://huggingface.co/datasets/PKU-Alignment/Align-Anything-Instruction-100K)           | Instruction-Following | 100K                            |


### Highlights  

Unlike existing datasets, which focus on individual modalities and vary in quality, **Align-Anything** offers consistent, high-quality data that encompasses **any modality (e.g., text, image, video and audio) in mixed inputs and outputs**. It provides detailed human preference annotations along with fine-grained language feedback for critique and refinement, enabling comprehensive evaluation and improvement across modalities.

![8](https://huggingface.co/datasets/PKU-Alignment/align-anything/resolve/main/new_distribution.png)

#### All-Modality Tasks

We present the combination of our **Align-Anything**, divided into three parts:
- **Any-to-Any** represents the bidirectional conversion of any type of input-output modality, such as text, video, audio and images.
- **Any-to-Text** represents the transition from non-textual inputs—such as image, video, and audio—into textual output.
- **Text-to-Any** represents the setting that text inputs are to be converted into any other modalities.


#### Fine-Grained Preference

How to Define a High-Quality Image? Assessing the quality of rich multimodal data is challenging with binary preferences on individual metrics. 

To address this, we have designed **Fine-Grained All-Modality Constitutions** to assist in annotating fine-grained preferences. These constitutions are composed of two main parts:

1. **General fine-grained metrics across modalities**, such as instruction-following, objective rules, clarity & aesthetics, information richness, and safety.
2. **Modality-specific constitutions**: For instance, for the video modality, we designed metrics such as temporal consistency, content coherence, and motion naturalness.

You can explore each modality’s subset dataset to view its fine-grained constitutions and definitions in detail.

According to the **Fine-Grained All-Modality Constitutions**, we utilized **GPT-4o**, **Gemini-1.5-Pro**, and **Human Crowds** to annotate data, resulting in comprehensive fine-grained annotations across all modalities.

#### Language Feedback

Multimodal data requires fine-grained annotations for better optimization. To guide the optimization process more effectively, **multimodal data** requires more fine-grained annotations. We propose a unified alignment method across all modalities by **utilizing language feedback**. Specifically, we provide critique and refinement feedback on each dimension as well as overall preferences for every data point. This feedback can be incorporated into your training process to enhance the performance of multimodal models.

### Annotation Pipeline

We demonstrate a multi-step process for refining AI responses based on multi-modal prompts. Raw prompts are refined based on specific modality and task, and then used to generate responses from various sources. Finally, we used the closed-source SOTA model and humans to perform cross-modality fine-grained annotation and language feedback to obtain the final dataset.

#### Collect Q-A Pairs

We start by designing specialized features tailored to various modalities. Based on specific modality tasks and their corresponding feature designs, we design **Fine-Grained All-Modality Constitutions**, according to which we refine the original prompts, which may initially be suboptimal, to create the final versions. We then collect responses from multiple sources, including self-constructed methods, the invocation of open-source and closed-source models, and human-generated answers.

#### Fine-grained Annotation
 
We conduct fine-grained preference annotations on the collected question-answer pairs. The annotations are sourced from both GPT-4, Gemini-1.5-Pro and human annotators. This annotation process covers a diverse range of dimensions, such as instruction-following, objective rules, aesthetics, information richness and safety, each with corresponding preferences and scoring criteria.

#### Language Feedback Generation

Finally, we provide language feedback on the responses. This involves determining the scope of critique, executing the critique, and providing refinement suggestions within the pipeline. This process captures both direct preferences for each modality and language-based feedback, ensuring a comprehensive evaluation and enhancement of the responses.

![Annotation Pipeline](https://github.com/D4YON3/images/blob/main/pipeline_annotation.png?raw=true)

### Datasets Comparison

> **Note**  
> Existing preference datasets are limited in scope and quality, focusing on specific modalities and lacking comprehensive annotations. In contrast, **Align-Anything** offers high-quality data across all modalities, with detailed human preference annotations and language feedback for critique and refinement. This comprehensive approach ensures a consistent evaluation and improvement of responses across modalities.

![Dataset Comparison](https://github.com/D4YON3/images/blob/main/dataset_compare.png?raw=true)

**Preference Annotation Methods** in the table consist of three parts, namely `Methods (A | S | F)` in the above table.


- **A** refers to the annotation source, which indicates how preferences are determined within the dataset. "Manual" denotes human annotation or manually constructed preferences, "Synthetic" refers to preferences generated or annotated by models like GPT-4V or other systems, and "Combined" refers to datasets aggregated from multiple sources.
- **S** represents the composition of preference signals, which may include scoring, ranking, and reasoning. In some cases, preferences are constructed by refining, correcting, or corrupting responses to form the desired preference pairs.
- **F** indicates whether the dataset provides fine-grained feedback at a more detailed level within those preference dimensions.

**Dimensions** indicate the primary preference challenges the dataset aims to address.

We compare the existing multimodal preference datasets, as shown in the table above. This comparison highlights the feedback diversity in our **Align-Anything**, which addresses the limitations of existing preference datasets, particularly following the expansion into multiple modalities.

### Human Agreement Analysis

We analyze the human agreement on the preference scores and the percentage of agreement on the preference scores. Our results show that the human agreement on the preference scores is high, indicating the reliability of the preference annotations. The percentage of agreement on the preference scores is also high, demonstrating the consistency of the preference annotations.


## Enhancing Agent Reasoning with Verifiable Rewards and RAGEN

### Key Features

1. **Verifiable Reward Integration**  
   We enable rule-based verifiable rewards to enhance LLMs' complex reasoning, particularly in mathematical reasoning and code generation. Unlike traditional reward models—which can be vulnerable to reward hacking and probabilistic errors—verifiable rewards use automated, rule-based checks (e.g., unit tests for code, answer matching in math) to provide deterministic and reliable feedback. This method improves training stability, reduces reward exploitation, and speeds up reward computation.

2. **Agent RL via RAGEN Framework**  
   We integrate the [RAGEN](https://github.com/RAGEN-AI/RAGEN) framework—originally implemented in [VeRL](https://github.com/volcengine/verl/tree/main)—into Align-Anything. RAGEN supports multi-turn interaction, memory retention, and sequential decision-making, enabling LLM agents to evolve through experience. The framework is tested across diverse environments including Bandit, Sokoban, Frozen Lake, and WebShop, providing a systematic way to study and improve agent behavior in stochastic and multi-step settings.

3. **Cross-Platform Training & Optimization**  
   The entire training and inference process can be conducted on Ascend 910B accelerators. We successfully ported and optimized the RAGEN training loop and verifiable reward modules, achieving performance consistent with NVIDIA A100 systems. Training stability was improved through hyperparameter tuning, enabling smooth and scalable RL training even in complex multi-turn settings.


### Experimental Results

We replicated experiments of [RAGEN](https://arxiv.org/abs/2504.20073v1) on Ascend 910B platforms:

- **Bandit Task**: Achieved 100% success rate within 200 training steps (40 minutes).

![image](https://hackmd.io/_uploads/SJ4tdyqYeg.png)


- **Sokoban Task**: Improved success rate from near-zero to 0.2 over 140–200 steps (2.3–4.3 hours), demonstrating stable reward growth and learning convergence.

![image](https://hackmd.io/_uploads/rkp9_JqKlg.png)
