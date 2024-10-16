# Overview

The evaluation module of align-anything supports a variety of multimodal benchmarks, such as `Text→Text`, `Text+Image/Video/Audio→Text`, and `Text→Image/Video/Audio`. For most modalities, we provide vLLM and Deepspeed as generation backends, allowing users to choose based on their devices and environments.  Since diffusion generation models for audio and video modalities primarily rely on different frameworks, we provide an option to load local files directly for evaluation in `Text→Image/Audio/Video` tasks (development completed and testing underway).

## Support Benchmarks

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Benchmark Table</title>
    <style>
        .scrollable-container {
            width: 100%;
            max-height: 75vh;
            overflow-y: auto;
            margin: 20px 0;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            text-align: center;
            table-layout: fixed;
        }
        th, td {
            padding: 10px;
            border-bottom: 1px solid #ddd;
        }
        th {
            font-weight: bold;
        }
        .content {
            padding: 5px;
            border-radius: 5px;
            display: inline-block;
        }
        a {
            text-decoration: none;
            color: inherit;
        }
        .star {
            color: red;
            font-size: 0.8em;
            vertical-align: super;
        }
        .note {
            margin-top: 10px;
            font-size: 0.9em;
            text-align: left;
        }
        .red-star {
            color: red;
        }
    </style>
</head>
<body>
    <div class="table-container">
        <div class="table-wrapper">
            <table>
                <tr>
                    <th>Benchmark</th>
                    <th>Modality</th>
                    <th>Support Backend</th>
                </tr>
                <tr>
                    <td><a href="https://huggingface.co/datasets/allenai/ai2_arc" target="_blank">ARC</a></td>
                    <td>Text→Text</td>
                    <td>vLLM</td>
                </tr>
                <tr>
                    <td><a href="https://huggingface.co/datasets/lukaemon/bbh" target="_blank">BBH</a></td>
                    <td>Text→Text</td>
                    <td>vLLM</td>
                </tr>
                <tr>
                    <td><a href="https://huggingface.co/datasets/facebook/belebele" target="_blank">Belebele</a></td>
                    <td>Text→Text</td>
                    <td>vLLM</td>
                </tr>
                <tr>
                    <td><a href="https://huggingface.co/datasets/haonan-li/cmmlu" target="_blank">CMMLU</a></td>
                    <td>Text→Text</td>
                    <td>vLLM</td>
                </tr>
                <tr>
                    <td><a href="https://huggingface.co/datasets/openai/gsm8k" target="_blank">GSM8K</a></td>
                    <td>Text→Text</td>
                    <td>vLLM</td>
                </tr>
                <tr>
                    <td><a href="https://huggingface.co/datasets/openai/openai_humaneval" target="_blank">HumanEval</a></td>
                    <td>Text→Text</td>
                    <td>vLLM</td>
                </tr>
                <tr>
                    <td><a href="https://huggingface.co/datasets/cais/mmlu" target="_blank">MMLU</a></td>
                    <td>Text→Text</td>
                    <td>vLLM</td>
                </tr>
                <tr>
                    <td><a href="https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro" target="_blank">MMLU-Pro</a></td>
                    <td>Text→Text</td>
                    <td>vLLM</td>
                </tr>
                <tr>
                    <td><a href="https://huggingface.co/datasets/HuggingFaceH4/mt_bench_prompts" target="_blank">MT-Bench</a></td>
                    <td>Text→Text</td>
                    <td>vLLM</td>
                </tr>
                <tr>
                    <td><a href="https://huggingface.co/datasets/google-research-datasets/paws-x" target="_blank">PAWS-X</a></td>
                    <td>Text→Text</td>
                    <td>vLLM</td>
                </tr>
                <tr>
                    <td><a href="https://huggingface.co/datasets/ehovy/race" target="_blank">RACE</a></td>
                    <td>Text→Text</td>
                    <td>vLLM</td>
                </tr>
                <tr>
                    <td><a href="https://huggingface.co/datasets/truthfulqa/truthful_qa" target="_blank">TruthfulQA</a></td>
                    <td>Text→Text</td>
                    <td>vLLM</td>
                </tr>
                <tr>
                    <td><a href="https://huggingface.co/datasets/HuggingFaceM4/A-OKVQA" target="_blank">A-OKVQA</a></td>
                    <td>Text+Image→Text</td>
                    <td>vLLM/DeepSpeed</td>
                </tr>
                <tr>
                    <td><a href="https://huggingface.co/datasets/lmms-lab/llava-bench-coco" target="_blank">LLaVA-Bench(COCO)</a></td>
                    <td>Text+Image→Text</td>
                    <td>vLLM</td>
                </tr>
                <tr>
                    <td><a href="https://huggingface.co/datasets/lmms-lab/llava-bench-in-the-wild" target="_blank">LLaVA-Bench(wild)</a></td>
                    <td>Text+Image→Text</td>
                    <td>vLLM</td>
                </tr>
                <tr>
                    <td><a href="https://huggingface.co/datasets/AI4Math/MathVista" target="_blank">MathVista</a></td>
                    <td>Text+Image→Text</td>
                    <td>vLLM/DeepSpeed</td>
                </tr>
                <tr>
                    <td><a href="https://huggingface.co/datasets/PKU-Alignment/MM-SafetyBench" target="_blank">MM-SafetyBench</a></td>
                    <td>Text+Image→Text</td>
                    <td>vLLM/DeepSpeed</td>
                </tr>
                <tr>
                    <td><a href="https://huggingface.co/datasets/lmms-lab/MMBench" target="_blank">MMBench</a></td>
                    <td>Text+Image→Text</td>
                    <td>vLLM/DeepSpeed</td>
                </tr>
                <tr>
                    <td><a href="https://huggingface.co/datasets/lmms-lab/MME" target="_blank">MME</a></td>
                    <td>Text+Image→Text</td>
                    <td>vLLM/DeepSpeed</td>
                </tr>
                <tr>
                    <td><a href="https://huggingface.co/datasets/MMMU/MMMU" target="_blank">MMMU</a></td>
                    <td>Text+Image→Text</td>
                    <td>vLLM</td>
                </tr>
                <tr>
                    <td><a href="https://huggingface.co/datasets/Lin-Chen/MMStar" target="_blank">MMStar</a></td>
                    <td>Text+Image→Text</td>
                    <td>vLLM/DeepSpeed</td>
                </tr>
                <tr>
                    <td><a href="https://huggingface.co/datasets/lmms-lab/MMVet" target="_blank">MMVet</a></td>
                    <td>Text+Image→Text</td>
                    <td>vLLM/DeepSpeed</td>
                </tr>
                <tr>
                    <td><a href="https://huggingface.co/datasets/lmms-lab/POPE" target="_blank">POPE</a></td>
                    <td>Text+Image→Text</td>
                    <td>vLLM/DeepSpeed</td>
                </tr>
                <tr>
                    <td><a href="https://huggingface.co/datasets/derek-thomas/ScienceQA" target="_blank">ScienceQA</a></td>
                    <td>Text+Image→Text</td>
                    <td>vLLM</td>
                </tr>
                <tr>
                    <td><a href="https://huggingface.co/datasets/sqrti/SPA-VL" target="_blank">SPA-VL</a></td>
                    <td>Text+Image→Text</td>
                    <td>vLLM/DeepSpeed</td>
                </tr>
                <tr>
                    <td><a href="https://huggingface.co/datasets/lmms-lab/textvqa" target="_blank">TextVQA</a></td>
                    <td>Text+Image→Text</td>
                    <td>vLLM</td>
                </tr>
                <tr>
                    <td><a href="https://huggingface.co/datasets/lmms-lab/VizWiz-VQA" target="_blank">VizWizVQA</a></td>
                    <td>Text+Image→Text</td>
                    <td>vLLM/DeepSpeed</td>
                </tr>
                <tr>
                    <td>
                        <a href="https://huggingface.co/datasets/qyang1021/AIR-Bench-Dataset" target="_blank">AIR-Bench</a><!-- <span class="star">*</span> -->
                    </td>
                    <td>Text+Audio→Text</td>
                    <td>DeepSpeed</td>
                </tr>
                <tr>
                    <td>
                        <a href="https://huggingface.co/datasets/OpenGVLab/MVBench" target="_blank">MVBench</a><!-- <span class="star">*</span> -->
                    </td>
                    <td>Text+Video→Text</td>
                    <td>vLLM</td>
                </tr>
                <tr>
                    <td>
                        <a href="https://huggingface.co/datasets/lmms-lab/Video-MME" target="_blank">Video-MME</a><!-- <span class="star">*</span> -->
                    </td>
                    <td>Text+Video→Text</td>
                    <td>vLLM</td>
                </tr>
                <tr>
                    <td>
                        <a href="https://huggingface.co/datasets/sayakpaul/coco-30-val-2014" target="_blank">COCO-val2014-30k</a><!-- <span class="star">*</span> -->
                    </td>
                    <td>Text→Image</td>
                    <td>Accelerate</td>
                </tr>
                <tr>
                    <td><a href="https://huggingface.co/datasets/zhwang/HPDv2" target="_blank">HPSv2</a></td>
                    <td>Text→Image</td>
                    <td>Accelerate</td>
                </tr>
                <tr>
                    <td><a href="https://huggingface.co/datasets/THUDM/ImageRewardDB" target="_blank">ImageReward</a></td>
                    <td>Text→Image</td>
                    <td>Accelerate</td>
                </tr>
                <tr>
                    <td>
                        <a href="https://huggingface.co/datasets/AudioLLMs/audiocaps_test" target="_blank">AudioCaps</a><!-- <span class="star">*</span> -->
                    </td>
                    <td>Text→Audio</td>
                    <td>Accelerate</td>
                </tr>
                <tr>
                    <td>
                        <a href="https://huggingface.co/datasets/BestWishYsh/ChronoMagic-Bench" target="_blank">ChronoMagic-Bench</a><!-- <span class="star">*</span> -->
                    </td>
                    <td>Text→Video</td>
                    <td>Accelerate</td>
                </tr>
            </table>
        </div>
    </div>
    <!-- <div class="note"><span class="red-star">*</span> Benchmarks with an asterisk are in test.</div> -->
</body>
</html>
