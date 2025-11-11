# Multi-turn Multimodal Pair Comparison Evaluation

This repository contains tools for evaluating dialogue models in multi-turn interaction tasks, supporting both global and local evaluation modes.

## Overview

The pair comparison evaluation system is designed to assess the quality of multi-turn multimodal conversations by comparing pairs of responses across different evaluation dimensions. The system uses LLM-as-a-judge approach to evaluate conversation quality.

## Evaluation Logic

### Core Concept
- **Pair Comparison**: For each conversation sample, two different responses (ResponseA and ResponseB) are compared
- **Random Swapping**: Response order is randomly swapped to avoid position bias
- **Multi-dimensional Evaluation**: Each response pair is evaluated across multiple quality dimensions
- **Statistical Analysis**: Results include accuracy, Pearson correlation coefficients, and confidence intervals

### Evaluation Modes

#### 1. Global Evaluation
Evaluates the overall quality of complete multi-turn conversations.

**Evaluation Categories:**
- `context_awareness`: Ability to maintain and understand dialogue history
- `helpfulness`: How well the model follows task instructions and provides complete solutions
- `crucial_step_recognition`: Accurate identification and completion of crucial steps
- `global_image_text_consistency`: Alignment between textual descriptions and generated images across turns
- `style_coherence`: Visual style consistency across generated images

#### 2. Local Evaluation
Evaluates the quality of individual turns within multi-turn conversations.

**Evaluation Categories:**
- `local_image_text_consistency`: Text-image alignment within a single turn
- `visual_perceptual_quality`: Visual realism and absence of artifacts in generated images
- `text_quality`: Clarity, coherence, and correctness of text output
- `context_coherence`: Logical consistency with prior dialogue context

## Features

- Support for both global and local evaluation modes
- Multiple evaluation metrics with detailed scoring criteria
- Comprehensive statistical analysis including Pearson correlation
- Detailed evaluation reports and data analysis
- Command-line interface for easy configuration
- Caching mechanism for efficient batch processing
- Ray-based parallel processing for scalability

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Using Command Line Scripts

```bash
# Run global evaluation
./run_pair_evaluation.sh -m gpt-4o -e global

# Run local evaluation
./run_pair_evaluation.sh -m gpt-4o -e local

# Run both global and local evaluation
./run_pair_evaluation.sh -m gpt-4o -e both
```

### Direct Python Usage

```python
from pair_inference import global_inference, local_inference

# Global evaluation
global_accuracy, global_pearson, global_results = global_inference(
    if_reason=True,
    evaluation_category=['context_awareness', 'helpfulness'],
    output_dir="./output",
    model_name="gpt-4o",
    temperature=0.5,
    limit=10  # Limit to 10 samples for testing
)

# Local evaluation
local_accuracy, local_pearson, local_results = local_inference(
    if_reason=True,
    evaluation_category=['text_quality', 'context_coherence'],
    output_dir="./output",
    model_name="gpt-4o",
    temperature=0.5,
    limit=10
)
```

## Command Line Parameters

- `-m, --model`: Model name to evaluate (default: gpt-4o)
- `-t, --temperature`: Generation temperature (default: 0.5)
- `-e, --mode`: Evaluation mode - global, local, or both (default: global)
- `-c, --categories`: Evaluation categories, comma-separated or "all" (default: auto-select based on mode)
- `-r, --reason`: Include evaluation reasoning (default behavior)
- `-n, --no-reason`: Exclude evaluation reasoning
- `-l, --limit`: Limit number of samples to process, 0 for all samples (default: 10)
- `-o, --output-dir`: Output directory (default: ./output)
- `-d, --cache-dir`: Cache directory (default: ./cache)

## Output Structure

Evaluation results are saved in the specified output directory with the following structure:

```
{output_dir}/{model}_with/no_reason_{timestamp}/
├── global/                      # Global evaluation results
│   ├── debug.json               # Detailed debug information
│   ├── clean.json               # Cleaned output results
│   ├── accuracy.json            # Accuracy and correlation results
│   ├── config_info.json         # Configuration information
│   └── summary.json             # Summary results
├── local/                       # Local evaluation results
│   ├── debug.json
│   ├── clean.json
│   ├── accuracy.json
│   ├── config_info.json
│   └── summary.json
└── combined_summary.json        # Combined results (when mode=both)
```

## Output Files Explained

### debug.json
Contains complete evaluation data including:
- Input prompts and system messages
- Raw model outputs
- Extracted preferences and reasons
- Ground truth comparisons
- Match analysis

### clean.json
Simplified version containing:
- Item IDs
- Model outputs
- Extracted preferences
- Ground truth labels
- Match results

### accuracy.json
Statistical analysis including:
- Accuracy scores per category
- Pearson correlation coefficients
- Confidence intervals
- Additional metrics (Kappa, F1)

### summary.json
High-level results:
- Overall accuracy
- Average Pearson coefficient
- Category-wise performance summary

## Usage Examples

1. **Evaluate single model on global tasks:**
   ```bash
   ./run_pair_evaluation.sh -m gpt-4o -e global
   ```

2. **Limit evaluation samples:**
   ```bash
   ./run_pair_evaluation.sh -m gpt-4o -e both -l 5
   ```

3. **Exclude evaluation reasoning:**
   ```bash
   ./run_pair_evaluation.sh -m gpt-4o -e global -n
   ```

4. **Specify particular evaluation metrics:**
   ```bash
   ./run_pair_evaluation.sh -m gpt-4o -e global -c "context_awareness,helpfulness"
   ```

5. **Custom output directory:**
   ```bash
   ./run_pair_evaluation.sh -m gpt-4o -e both -o "./my_results"
   ```

## Technical Details

### Data Processing
- **Vision Input Handling**: Supports various image formats (base64, URLs, local paths)
- **Content Processing**: Converts multimodal content to API-compatible format
- **Random Response Swapping**: Ensures unbiased evaluation by randomizing response order

### Evaluation Pipeline
1. **Data Loading**: Load test datasets with conversation history and response pairs
2. **Prompt Generation**: Create evaluation prompts with appropriate system instructions
3. **API Calls**: Batch processing with caching and retry mechanisms
4. **Response Parsing**: Extract preferences and reasoning from model outputs
5. **Statistical Analysis**: Calculate accuracy, correlation, and confidence metrics

### Preference Extraction
The system uses sophisticated regex patterns to extract preferences from various output formats:
- Boxed notation: `\boxed{ResponseA}` or `\boxed{ResponseB}`
- Bracketed lists: `[[category, reason, \boxed{ResponseA}]]`
- Multiple format detection and fallback mechanisms

## Configuration

### API Settings
Configure API access in `config.py`:
```python
API_KEY = 'your-api-key'
API_BASE_URL = 'https://your-api-endpoint.com'
MAX_RETRIES = 3
NUM_WORKERS = 20
```

### Model Parameters
```python
MODEL_NAME = 'gpt-4o'
TEMPERATURE = 0.5
```

### Test Data Paths
```python
GLOBAL_TEST_FILE = './test_dataset/pair_comparison_global_test_data_300_supple_2.json'
LOCAL_TEST_FILE = './test_dataset/pair_comparison_local_test_data_900_supple_2.json'
```

## Module Structure

- `pair_inference.py`: Main inference logic and evaluation pipeline
- `pair_data_loader.py`: Data loading and preprocessing utilities
- `api_utils.py`: API calling utilities with caching and retry mechanisms
- `config.py`: Configuration settings and constants
- `system_prompt/`: System prompt templates for different evaluation modes
  - `pair_eval_global_judge.py`: Global evaluation prompts and criteria
  - `pair_eval_local_judge.py`: Local evaluation prompts and criteria

## Performance and Scalability

- **Parallel Processing**: Uses Ray for efficient batch API calls
- **Caching System**: Avoids redundant API calls through intelligent caching
- **Memory Efficient**: Processes large datasets without memory overflow
- **Retry Mechanisms**: Robust handling of API failures and rate limits

## Statistical Analysis

The system provides comprehensive statistical analysis:
- **Accuracy**: Simple match rate between model and ground truth preferences
- **Pearson Correlation**: Measures linear relationship strength
- **Confidence Intervals**: 95% confidence intervals for correlation coefficients
- **Cohen's Kappa**: Inter-rater agreement measure
- **F1 Score**: Binary classification performance metric

## License

MIT License

