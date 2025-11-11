# Evaluation Configuration Settings

# Global evaluation categories for multi-turn conversations
GLOBAL_EVALUATION_CATEGORIES = ['context_awareness', 'helpfulness', 'crucial_step_recognition', 'global_image_text_consistency', 'style_coherence']

# Local evaluation categories for individual responses
LOCAL_EVALUATION_CATEGORIES = ['local_image_text_consistency', 'visual_perceptual_quality', 'text_quality', 'context_coherence']

# Default model configuration
MODEL_NAME = 'gpt-4o'

# Generation temperature (0.0 to 1.0)
TEMPERATURE = 0.5

# API configuration - set your credentials here
API_KEY = ''
API_BASE_URL = ''

# Request configuration
MAX_RETRIES = 3
NUM_WORKERS = 20

# Default test dataset path
TEST_FILE = './test_dataset/500_crucial_step_test_data.json'
