# Crucial Step Recognition Evaluation System

This system evaluates a model's ability to identify and recognize crucial steps in multi-turn conversations, particularly in scenarios where models need to guide users through complex, step-by-step processes.

## Quick Start

For a quick demonstration of the system, run:

```bash
python quick_example.py
```

This will:
- Check your configuration
- Show sample data structure  
- Run evaluation on 5 samples
- Display results summary

For more detailed usage, continue reading below.

## Overview

The Crucial Step Recognition evaluation measures how well a model can:
- Identify essential steps in a multi-turn dialogue
- Distinguish between crucial and optional information
- Maintain logical sequence and flow in step-by-step guidance
- Avoid providing irrelevant or incorrect information that could mislead users

## System Architecture

### Core Components

1. **`config.py`** - Configuration settings for API endpoints, models, and evaluation parameters
2. **`system_prompt.py`** - Contains evaluation prompts for both inference and judging
3. **`data_loader.py`** - Handles loading and processing of test data
4. **`api_utils.py`** - API utilities for making batch requests with caching
5. **`recognition_inference.py`** - Main inference and evaluation script
6. **`run_recognition_evaluation.sh`** - Batch evaluation script for multiple models

### Data Structure

The evaluation uses JSON test data with the following structure:
```json
{
  "id": 0,
  "conversations": [
    {
      "round": 1,
      "prompt": "User question",
      "prompt_images": ["image_path"],
      "response": "Model response",
      "response_images": ["image_path"]
    }
  ],
  "annotation": {
    "reason_crucial_step_recognition": "Ground truth explanation"
  }
}
```

## Setup and Configuration

### 1. Environment Requirements

```bash
pip install ray urllib3 pillow tqdm qwen-vl-utils
```

### 2. Configuration

Edit `config.py` to set your API credentials:

```python
API_KEY = 'your_api_key_here'
API_BASE_URL = 'your_api_endpoint_here'
MODEL_NAME = 'gpt-4o'  # Default model
TEMPERATURE = 0.5
MAX_RETRIES = 3
NUM_WORKERS = 20
```

### 3. Test Data

Place your test data in the `test_dataset/` directory. The default path is:
```
test_dataset/500_crucial_step_test_data.json
```

## Usage

### Single Model Evaluation

Run evaluation for a single model:

```bash
python recognition_inference.py \
    --inference-model "gpt-4" \
    --judge-model "gpt-4o" \
    --temperature 0.5 \
    --test-file "test_dataset/500_crucial_step_test_data.json" \
    --output-dir "./output" \
    --cache-dir "./cache" \
    --limit 100 \
    --verbose
```

#### Parameters:
- `--inference-model`: Model to evaluate for crucial step recognition
- `--judge-model`: Model used to judge the quality of inference results
- `--temperature`: Generation temperature (0.0 to 1.0)
- `--test-file`: Path to test dataset JSON file
- `--output-dir`: Directory to save evaluation results
- `--cache-dir`: Directory for API response caching
- `--limit`: Number of samples to evaluate (0 = all samples)
- `--verbose`: Show detailed debugging information

### Batch Model Evaluation

Run evaluation for multiple models automatically:

```bash
bash run_recognition_evaluation.sh
```

This script will:
- Evaluate multiple models defined in the `INFERENCE_MODELS` array
- Generate timestamped output directories
- Create comparison tables and summaries
- Show progress bars and time estimates
- Save comprehensive evaluation reports

#### Customizing Batch Evaluation

Edit the script to modify:

```bash
# Models to evaluate
declare -a INFERENCE_MODELS=(
    "claude-3-7-sonnet-20250219-thinking"
    "o4-mini-2025-04-16"
    "gpt-4.1-2025-04-14"
    "gpt-4o"
)

# Evaluation settings
JUDGE_MODEL="gpt-4o"
TEMPERATURE=0.7
SAMPLE_LIMIT=100  # 0 for all samples
```

## Output Structure

### Individual Model Results

Each evaluation creates a timestamped directory with:

```
output/
└── crucial_step_{model_name}_{timestamp}/
    ├── inference_results.json      # Raw model inferences
    ├── evaluation_results.json     # Judged results with scores
    └── statistics.json            # Performance statistics
```

### Batch Evaluation Results

```
batch_recognition_logs/
├── {timestamp}/
│   ├── {model1}/
│   │   ├── inference_results.json
│   │   ├── evaluation_results.json
│   │   └── statistics.json
│   ├── {model2}/
│   │   └── ...
│   ├── evaluation_summary.txt      # Cross-model comparison
│   └── run_log.txt                # Execution log
├── latest -> {timestamp}/          # Symlink to latest results
└── cache_{timestamp}/              # API response cache
```

## Evaluation Metrics

### Scoring System

Models are evaluated on a 1-5 scale:

- **5**: Flawless identification of crucial steps, perfect accuracy and completeness
- **4**: Almost correct with minor inaccuracies that don't affect overall meaning
- **3**: Partially accurate, captures some key steps but misses critical elements
- **2**: Misses most important steps, contains numerous errors
- **1**: Fails to capture essential steps, major errors or misinterpretations

### Key Evaluation Criteria

1. **Step Identification**: Ability to identify essential vs. optional steps
2. **Logical Sequence**: Maintaining proper order of operations
3. **Completeness**: Including all necessary information
4. **Accuracy**: Correctness of identified steps
5. **Clarity**: Clear communication of step relationships

## Example Evaluation Scenarios

### Good Crucial Step Recognition
```
Task: Drawing a cat step-by-step
✓ Crucial steps: Sketch outline → Refine facial features → Adjust proportions → Apply color
✓ Logical progression maintained
✓ All essential steps included
```

### Poor Crucial Step Recognition
```
Task: Drawing a cat step-by-step
✗ Asks user to color before outline is drawn
✗ Skips essential proportion adjustment
✗ Provides irrelevant details about cat breeds
```

## Performance Analysis

### Statistics Output

```json
{
  "average_score": 3.85,
  "score_distribution": {
    "1": 5,
    "2": 12,
    "3": 28,
    "4": 35,
    "5": 20
  },
  "total_evaluated": 100,
  "total_samples": 100,
  "config": {
    "inference_model": "gpt-4",
    "judge_model": "gpt-4o",
    "temperature": 0.5
  }
}
```

### Comparison Table

The system generates comparative tables showing:
- Model rankings by average score
- Sample counts and success rates
- Performance distributions
- Best performing model identification

## Troubleshooting

### Common Issues

1. **API Rate Limiting**: Adjust `MAX_RETRIES` and `NUM_WORKERS` in config
2. **Memory Issues**: Reduce `NUM_WORKERS` or process smaller batches
3. **Cache Corruption**: Delete cache directory and restart
4. **Missing Dependencies**: Install required packages with pip

### Debug Mode

Use `--verbose` flag to see detailed processing information:
- Sample-by-sample score extraction
- API response parsing details
- Error messages and stack traces

## Extending the System

### Adding New Models

1. Add model name to `INFERENCE_MODELS` array in batch script
2. Ensure API endpoint supports the model
3. Adjust any model-specific parameters in config

### Custom Evaluation Criteria

Modify `system_prompt.py` to adjust:
- Scoring criteria and ranges
- Evaluation focus areas
- Judge prompt instructions

### Data Format Support

Extend `data_loader.py` to support:
- Different input formats
- Additional metadata fields
- Custom conversation structures

## Performance Optimization

### Caching

The system uses intelligent caching to avoid redundant API calls:
- Responses cached by content hash
- Automatic cache hit detection
- Configurable cache directories

### Parallel Processing

Ray-based parallel processing enables:
- Concurrent API requests
- Efficient resource utilization
- Progress tracking and monitoring

## License

This evaluation system is designed for research and development purposes in AI model assessment. 