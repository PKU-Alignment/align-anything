"""
Configuration file
"""

# API Key
API_KEY = ""  # Replace with your actual key
API_BASE_URL = "" # Replace with your actual base url

# Maximum retry count
MAX_RETRIES = 3

# Number of parallel worker threads
NUM_WORKERS = 4

# Test dataset path
TEST_FILE = './test_dataset/score_evaluation_test.json'

# Model configuration
MODEL_NAME = "gpt-4o"  # Model name to use
TEMPERATURE = 0.2      # Generation temperature

# Evaluation configuration
# Evaluation categories list, options: ['context_awareness', 'helpfulness', 'crucial_step_recognition', 'global_image_text_consistency', 'style_coherence']
# Set to ['all'] to include all categories
GLOBAL_EVALUATION_CATEGORIES = ['context_awareness', 'helpfulness', 'crucial_step_recognition', 'global_image_text_consistency', 'style_coherence']
LOCAL_EVALUATION_CATEGORIES = ['local_image_text_consistency', 'visual_perceptual_quality', 'text_quality', 'context_coherence']
# Whether to include reasoning
INCLUDE_REASON = True

# Output directory
OUTPUT_DIR = "./output"

# Cache directory
CACHE_DIR = "./cache" 