# Align-Anything-100K: Text-to-Text Instruction Following Dataset

[[🏠 Homepage](https://github.com/PKU-Alignment/align-anything)]
[[🤗 Instruction-Dataset-100K(en)](https://huggingface.co/datasets/PKU-Alignment/Align-Anything-Instruction-100K)]
[[🤗 Instruction-Dataset-100K(zh)](https://huggingface.co/datasets/PKU-Alignment/Align-Anything-Instruction-100K-zh)]


##  Highlights

<div class="col-md-12"> 
      <ul>
          <li><b>Data sources:</b> 
              <a href="https://huggingface.co/datasets/PKU-Alignment/PKU-SafeRLHF-QA" target="_blank">PKU-SafeRLHF QA</a> , 
              <a href="https://huggingface.co/datasets/knkarthick/dialogsum" target="_blank">DialogSum</a>, 
              <a href="https://ai.meta.com/research/publications/towards-empathetic-open-domain-conversation-models-a-new-benchmark-and-dataset" target="_blank">Empathetic</a>, 
              <a href="https://github.com/XueFuzhao/InstructionWild" target="_blank">Instruction-Wild</a>, 
              and <a href="https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json" target="_blank">Alpaca</a>. </li>
          <li><b>100K QA pairs:</b> By leveraging GPT-4 to annotate meticulously refined instructions, we obtain 105,333 QA pairs. </li>   
      </ul>  
</div>

## Dataset Summary

This dataset is a sibling project of [Align-Anything](https://github.com/PKU-Alignment/align-anything).

We provide a high-quality instruction-following dataset consisting of 100K question-answer entries, annotated and refined by GPT-4. Our prompts are sourced from multiple public datasets such as [PKU-SafeRLHF Dataset QA](https://huggingface.co/datasets/PKU-Alignment/PKU-SafeRLHF-QA), [DialogSum](https://huggingface.co/datasets/knkarthick/dialogsum), [Empathetic Dataset](https://ai.meta.com/research/publications/towards-empathetic-open-domain-conversation-models-a-new-benchmark-and-dataset), [Alpaca](https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json), and [InstructionWild](https://github.com/XueFuzhao/InstructionWild). Each prompt is refined by GPT-4 under expert demonstration and specific guidelines, followed by GPT-4's annotation of the responses. This comprehensive and fine-grained pipeline results in a high-quality instruction-following dataset.

## Dataset Comparison

### Detailed Results

We visualize our prompt distribution and compared it with the widely-used instruction-following dataset, [Alpaca-52K](https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json). Our dataset covers a broader range of prompt types and includes various task types such as text summarization, sentiment analysis, etc.

<div align="center">
  <img src="https://huggingface.co/datasets/PKU-Alignment/Align-Anything-Instruction-100K/resolve/main/vs.png" width="70%"/>
</div>

We train several base models using both Align-Anything-Instruction-100K (sampled 52K) and Alpaca-52K. We evaluate the fine-tuned models on the [Just-Eval](https://huggingface.co/datasets/re-align/just-eval-instruct) benchmark, assessing the responses across five dimensions: helpfulness, clarity, factuality, depth, and engagement. The models demonstrate excellent performance in all dimensions.

<div align="center">
  <img src="https://huggingface.co/datasets/PKU-Alignment/Align-Anything-Instruction-100K/resolve/main/performance.png" width="70%"/>
</div>

## Evaluation Details
### Just-Eval Overview

[Just-Eval](https://huggingface.co/datasets/re-align/just-eval-instruct) covers multiple prompts that fully assess the model's instruction-following capabilities, such as [AlpacaEval](https://huggingface.co/datasets/tatsu-lab/alpaca_eval), [LIMA-test](https://huggingface.co/datasets/GAIR/lima/viewer/plain_text/test), [MT-bench](https://huggingface.co/datasets/HuggingFaceH4/mt_bench_prompts), [Anthropic red-teaming](https://huggingface.co/datasets/Anthropic/hh-rlhf/tree/main/red-team-attempts), and [MaliciousInstruct](https://github.com/Princeton-SysML/Jailbreak_LLM/blob/main/data/MaliciousInstruct.txt). 

We utilize the 800 instructions that focus on problem-solving tests without considering the safety of responses, following the benchmark guidelines outlined [here](https://allenai.github.io/re-align/just_eval.html).


### Evaluation Criterias

We adopt the same evaluation criteria as the [JustEval Benchmark](https://allenai.github.io/re-align/index.html), detailed as follows:

<div class="col-md-12"> 
    <ul>
        <li><b>Helpfulness:</b> Evaluates how well the response addresses the given query or question and assists the user. A good response is highly relevant and helpful.</li>
        <li><b>Clarity:</b> Assesses the logical flow and coherence of the response. A good response is well-structured, with ideas presented clearly and coherently.</li>
        <li><b>Factuality:</b> Assesses the accuracy of the information presented in the response. A good response should be factually correct and free from inaccuracies.</li>
        <li><b>Depth:</b> Evaluates the thoroughness and detail of the response. A good response should be comprehensive and in-depth.</li>
        <li><b>Engagement:</b> Assesses how engaging and natural the response sounds in a conversational context. A good response should feel engaging and have a human-like tone.</li>
    </ul>  
</div>


## Usage

To load our dataset, use the `load_dataset()` function as follows:

```python
from datasets import load_dataset

dataset = load_dataset("PKU-Alignment/Align-Anything-Instruction-100K")
```
