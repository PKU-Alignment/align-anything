# Multi-turn Multimodal Score Evaluation 

This tool is designed to evaluate AI model performance in multi-turn multimodal communications, supporting score extraction, reasoning analysis, and consistency calculation with ground truth annotations.

## Features

- Support for batch inference and evaluation
- Extract scores and reasoning from model outputs
- Calculate accuracy and Pearson correlation coefficient between model scores and ground truth
- Support for custom evaluation categories and configurations
- Handle complete context of multi-turn dialogues
- Two evaluation modes: Global (overall conversation) and Local (per-turn)

## Installation

Install the required Python dependencies:

```bash
pip install numpy scipy ray tqdm pillow urllib3 qwen_vl_utils
```

## Configuration

Before running evaluations, you can modify configurations in `utils/config.py` or use command line arguments:

- `TEST_FILE`: Path to test dataset
- `MODEL_NAME`: Model name to use
- `TEMPERATURE`: Generation temperature
- `GLOBAL_EVALUATION_CATEGORIES`: Global evaluation categories
- `LOCAL_EVALUATION_CATEGORIES`: Local evaluation categories  
- `INCLUDE_REASON`: Whether to include reasoning
- `OUTPUT_DIR`: Output directory
- `CACHE_DIR`: Cache directory

### Evaluation Categories

**Global Evaluation Categories** (for overall conversation assessment):
- `context_awareness`: Model's ability to maintain and understand dialogue history
- `helpfulness`: How well the model follows instructions and provides complete information
- `crucial_step_recognition`: Ability to identify and complete key steps in multi-turn interactions
- `global_image_text_consistency`: Alignment between text descriptions and generated images across the conversation
- `style_coherence`: Visual style consistency across generated images

**Local Evaluation Categories** (for individual turn assessment):
- `local_image_text_consistency`: Text-image alignment within a single turn
- `visual_perceptual_quality`: Visual realism and absence of artifacts in generated images
- `text_quality`: Clarity, coherence, and correctness of text output
- `context_coherence`: Logical continuation from previous dialogue context

## Usage

### Basic Usage

```bash
python score_inference.py
```

### Command Line Arguments

```bash
# Show help
python score_inference.py -h

# Specify model and temperature
python score_inference.py --model gpt-4o --temperature 0.2

# Choose evaluation mode
python score_inference.py --mode global          # Global evaluation only
python score_inference.py --mode local           # Local evaluation only  
python score_inference.py --mode both            # Both modes

# Specify evaluation categories
python score_inference.py --categories context_awareness,helpfulness
python score_inference.py --categories all       # All categories for the mode

# Include/exclude reasoning
python score_inference.py --reason               # Include reasoning
python score_inference.py --no-reason            # Exclude reasoning

# Limit sample count
python score_inference.py --limit 10             # Process only first 10 samples

# Specify files and directories
python score_inference.py --test-file /path/to/test.json --output-dir ./my_output

# Complete example
python score_inference.py --model gpt-4o --temperature 0.2 --mode both --categories all --reason --limit 5
```

### Parameter Reference

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--model` | Model name to use | From config |
| `--temperature` | Generation temperature | From config |
| `--mode` | Evaluation mode: global/local/both | global |
| `--categories` | Evaluation categories (comma-separated) or "all" | From config |
| `--reason` | Include reasoning in evaluation | From config |
| `--no-reason` | Exclude reasoning | Opposite of config |
| `--test-file` | Test dataset path | From config |
| `--output-dir` | Output directory | From config |
| `--cache-dir` | Cache directory | From config |
| `--limit` | Limit number of samples (0 = all) | 0 |

## Output Structure

Results are saved in `OUTPUT_DIR/MODEL_NAME_[with/no]_reason_timestamp/` directory:

### Global Evaluation Mode (`global/`)
- `debug.json`: Complete debugging data with all intermediate results
- `clean.json`: Clean output results without debugging information
- `accuracy.json`: Accuracy and Pearson coefficient for each category
- `summary.json`: Overall model performance summary
- `config_info.json`: Evaluation configuration details

### Local Evaluation Mode (`local/`)
- Same structure as global evaluation but for per-turn analysis

### Both Modes (`combined_summary.json`)
- Comprehensive summary when both modes are executed

## Evaluation Metrics

- **Accuracy**: Percentage of model scores that match ground truth annotations
- **Pearson Correlation Coefficient**: Measures linear correlation between model scores and ground truth
- **Per-Category Analysis**: Individual metrics for each evaluation dimension

## Score Extraction

The tool uses sophisticated regex patterns to extract scores from model outputs in various formats:
- `[category, reason, \boxed{score}]`
- `[[category, reason, \boxed{score}]]`
- Multiple boxed notation variations: `\boxed{N}`, `\\boxed{N}`, etc.

## API Configuration

Configure your API settings in `utils/config.py`:
```python
API_KEY = "your-api-key-here"
API_BASE_URL = "https://your-api-endpoint.com"
```

## Caching

The tool implements intelligent caching to avoid redundant API calls:
- Cache files are stored in the specified cache directory
- Cached results are reused for identical requests
- Cache files use SHA256 hashes of request parameters

## System Architecture

```
score_evaluation/
├── score_inference.py          # Main evaluation script
├── data_loader.py             # Data loading and processing
├── api_utils.py               # API utilities and caching
├── utils/
│   └── config.py              # Configuration settings
├── system_prompt/
│   ├── score_eval_global_judge.py   # Global evaluation prompts
│   └── score_eval_local_judge.py    # Local evaluation prompts
└── test_dataset/              # Test data directory
```

## Examples

### Evaluate with Global Categories Only
```bash
python score_inference.py --mode global --categories context_awareness,helpfulness --reason
```

### Evaluate Each Turn Locally
```bash
python score_inference.py --mode local --categories all --limit 20
```

### Full Evaluation with Both Modes
```bash
python score_inference.py --mode both --categories all --reason --model gpt-4o
```

## Troubleshooting

**"Constant Array" Pearson Coefficient**: This indicates that either model scores or ground truth contain constant values (all same score), making correlation calculation impossible.

**Score Extraction Failures**: The tool supports multiple output formats. If extraction fails, check that your model output follows one of the supported formats with `\boxed{score}` notation.

**API Errors**: Verify your API key and endpoint in the configuration file. Check network connectivity and API rate limits.

## Recent Updates

- **English Translation**: All Chinese text translated to English
- **Enhanced Score Extraction**: Improved regex patterns for various output formats
- **Dual Evaluation Modes**: Support for both global and local evaluation
- **Comprehensive CLI**: Full command-line argument support
- **Intelligent Caching**: Efficient caching mechanism to reduce API costs
- **Detailed Output**: Comprehensive result files with debugging information 