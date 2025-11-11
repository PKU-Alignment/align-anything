GLOBAL_EVALUATION_CATEGORIES = ['context_awareness', 'helpfulness', 'crucial_step_recognition', 'global_image_text_consistency', 'style_coherence']
LOCAL_EVALUATION_CATEGORIES = ['local_image_text_consistency', 'visual_perceptual_quality', 'text_quality', 'context_coherence']

MODEL_NAME = 'gpt-4o'

TEMPERATURE = 0.5

GLOBAL_TEST_FILE = './test_dataset/global/pair_comparison_global_test_data_300.json'

LOCAL_TEST_FILE = './test_dataset/local/pair_comparison_local_test_data_900.json'

API_KEY = '' # your api key
API_BASE_URL = '' # your api base url
MAX_RETRIES = 3

NUM_WORKERS = 20
