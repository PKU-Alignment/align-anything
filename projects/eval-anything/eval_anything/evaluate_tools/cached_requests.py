# Copyright 2025 PKU-Alignment Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""
Cached requests tool for making API calls with caching support.
Adapted for eval-anything framework.
"""

import json
import os
import time
from typing import Any, Dict, List, Optional

import requests

from eval_anything.evaluate_tools.base_tools import BaseTool
from eval_anything.utils.uuid import UUIDGenerator


class CachedRequestsTool(BaseTool):
    """
    Tool for making API requests with caching to avoid duplicate calls.
    Inherits from BaseTool to integrate with eval-anything framework.
    """

    def __init__(self, cache_dir: str = './cache', max_try: int = 3, timeout: int = 3600, **kwargs):
        """
        Initialize the cached requests tool.

        Args:
            cache_dir: Directory to store cache files
            max_try: Maximum number of retry attempts
            timeout: Request timeout in seconds
            **kwargs: Additional arguments passed to BaseTool
        """
        super().__init__(**kwargs)
        self.cache_dir = cache_dir
        self.max_try = max_try
        self.timeout = timeout
        self.uuid_generator = UUIDGenerator()

        # Ensure cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)

    def apply(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        max_completion_tokens: int = 4096,
        temperature: float = 0.7,
        repetition_penalty: float = 1.0,
        top_p: float = 0.9,
        api_key: Optional[str] = None,
        api_base: str = None,
        **kwargs,
    ) -> str:
        """
        Make API requests with caching to avoid duplicate calls.

        Args:
            messages: List of message dictionaries to send to the API
            model: Model name to use for completion
            max_completion_tokens: Maximum number of tokens in the completion
            temperature: Sampling temperature (higher = more random)
            repetition_penalty: Penalty for token repetition
            top_p: Nucleus sampling parameter
            api_key: API key for authentication (if empty, will check environment variables)
            api_base: Base URL for API endpoint (if empty, will check environment variables)
            **kwargs: Additional parameters

        Returns:
            The completed text from the API response
        """
        return self.cached_requests(
            messages=messages,
            model=model,
            max_completion_tokens=max_completion_tokens,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            top_p=top_p,
            api_key=api_key,
            api_base=api_base,
        )

    def cached_requests(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        max_completion_tokens: int = 4096,
        temperature: float = 0.7,
        repetition_penalty: float = 1.0,
        top_p: float = 0.9,
        api_key: Optional[str] = None,
        api_base: str = None,
    ) -> str:
        """
        Make API requests with caching to avoid duplicate calls.

        Args:
            messages: List of message dictionaries to send to the API
            model: Model name to use for completion
            max_completion_tokens: Maximum number of tokens in the completion
            temperature: Sampling temperature (higher = more random)
            repetition_penalty: Penalty for token repetition
            top_p: Nucleus sampling parameter
            api_key: API key for authentication (if empty, will check environment variables)
            api_base: Base URL for API endpoint (if empty, will check environment variables)

        Returns:
            The completed text from the API response
        """

        if not api_key:
            api_key = os.environ.get('API_KEY', '')

        if not api_base:
            api_base = os.environ.get('API_BASE', 'https://api.openai.com/v1')

        if not api_key:
            raise ValueError('API key is not provided')

        # Generate cache key using eval-anything's UUID generator
        cache_key_data = {
            'messages': messages,
            'max_completion_tokens': max_completion_tokens,
            'temperature': temperature,
            'repetition_penalty': repetition_penalty,
            'top_p': top_p,
            'model': model,
        }
        uuid = self.uuid_generator(cache_key_data)

        cache_path = os.path.join(self.cache_dir, f'{uuid}.json')

        # Check if cached result exists
        if os.path.exists(cache_path):
            try:
                with open(cache_path, encoding='utf-8') as f:
                    result = json.load(f)
                    self.logger.log('info', f'Retrieved cached response from {cache_path}')
                    return result
            except json.JSONDecodeError:
                self.logger.log('warning', f'Invalid cache file {cache_path}, removing it')
                os.remove(cache_path)

        # Make API request with retries
        max_try = self.max_try
        while max_try > 0:
            try:
                headers = {'Content-Type': 'application/json', 'Connection': 'close'}
                if api_key:
                    headers['Authorization'] = f'Bearer {api_key}'

                response = requests.post(
                    api_base,
                    headers=headers,
                    json={
                        'model': model,
                        'max_completion_tokens': max_completion_tokens,
                        'messages': messages,
                        'temperature': temperature,
                        'top_p': top_p,
                        'repetition_penalty': repetition_penalty,
                    },
                    timeout=self.timeout,
                )

                if response.status_code == 200:
                    response_text = response.json()['choices'][0]['message']['content']

                    # Cache the response
                    with open(cache_path, 'w', encoding='utf-8') as f:
                        json.dump(response_text, f, ensure_ascii=False)

                    self.logger.log(
                        'info', f'API request successful, cached response to {cache_path}'
                    )
                    return response_text
                else:
                    if response.status_code == 400:
                        error_detail = response.json()['error']['message']
                        err_msg = f'API error, status code: {response.status_code}\nresponse: {error_detail}'
                    else:
                        err_msg = f'API error, status code: {response.status_code}, error: {response.text}'
                    self.logger.log('error', err_msg)
            except Exception as e:
                self.logger.log('error', f'Request failed: {str(e)}')

            time.sleep(3)
            max_try -= 1

        self.logger.log('error', 'API failed after all retries')
        return ''


def cached_requests(
    messages: List[Dict[str, Any]],
    model: str,
    max_completion_tokens: int = 4096,
    temperature: float = 0.7,
    repetition_penalty: float = 1.0,
    top_p: float = 0.9,
    api_key: Optional[str] = None,
    api_base: str = None,
    cache_dir: str = './cache',
    max_try: int = 3,
    timeout: int = 3600,
) -> str:
    """
    Convenience function for making cached API requests.
    Creates a CachedRequestsTool instance and calls it.

    Args:
        messages: List of message dictionaries to send to the API
        model: Model name to use for completion
        max_completion_tokens: Maximum number of tokens in the completion
        temperature: Sampling temperature (higher = more random)
        repetition_penalty: Penalty for token repetition
        top_p: Nucleus sampling parameter
        api_key: API key for authentication (if empty, will check environment variables)
        api_base: Base URL for API endpoint (if empty, will check environment variables)
        cache_dir: Directory to store cache files
        max_try: Maximum number of retry attempts
        timeout: Request timeout in seconds

    Returns:
        The completed text from the API response
    """
    tool = CachedRequestsTool(cache_dir=cache_dir, max_try=max_try, timeout=timeout)

    return tool.cached_requests(
        messages=messages,
        model=model,
        max_completion_tokens=max_completion_tokens,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        top_p=top_p,
        api_key=api_key,
        api_base=api_base,
    )


def test_cached_requests_tool():
    """
    Simple test function for the CachedRequestsTool.
    Tests basic functionality including caching behavior.
    """
    import shutil
    import tempfile

    print('Testing CachedRequestsTool...')

    # Create a temporary cache directory for testing
    test_cache_dir = tempfile.mkdtemp(prefix='cached_requests_test_')

    try:
        # Test 1: Basic tool initialization
        print('Test 1: Tool initialization...')
        tool = CachedRequestsTool(cache_dir=test_cache_dir, max_try=1, timeout=10)
        assert tool.cache_dir == test_cache_dir
        assert tool.max_try == 1
        assert tool.timeout == 10
        print('✓ Tool initialization successful')

        # Test 2: UUID generation consistency
        print('Test 2: UUID generation consistency...')
        messages = [{'role': 'user', 'content': 'Hello, how are you?'}]
        model = 'gpt-3.5-turbo'

        # Generate UUID twice with same parameters
        cache_key_data = {
            'messages': messages,
            'max_completion_tokens': 100,
            'temperature': 0.7,
            'repetition_penalty': 1.0,
            'top_p': 0.9,
            'model': model,
        }
        uuid1 = tool.uuid_generator(cache_key_data)
        uuid2 = tool.uuid_generator(cache_key_data)
        assert uuid1 == uuid2, 'UUIDs should be consistent for same parameters'
        print(f'✓ UUID generation consistent: {uuid1}')

        # Test 3: Cache directory creation
        print('Test 3: Cache directory creation...')
        assert os.path.exists(test_cache_dir), 'Cache directory should exist'
        print('✓ Cache directory created successfully')

        # Test 4: Test without API key (should raise ValueError)
        print('Test 4: API key validation...')
        # Clear environment variables for this test
        original_api_key = os.environ.get('API_KEY')
        original_api_base = os.environ.get('API_BASE')
        if 'API_KEY' in os.environ:
            del os.environ['API_KEY']
        if 'API_BASE' in os.environ:
            del os.environ['API_BASE']

        try:
            tool.apply(messages=messages, model=model)
            assert False, 'Should raise ValueError when no API key is provided'
        except ValueError as e:
            assert 'API key is not provided' in str(e)
            print('✓ API key validation working correctly')
        finally:
            # Restore environment variables
            if original_api_key:
                os.environ['API_KEY'] = original_api_key
            if original_api_base:
                os.environ['API_BASE'] = original_api_base

        # Test 5: Cache file operations (mock)
        print('Test 5: Cache file operations...')
        cache_path = os.path.join(test_cache_dir, f'{uuid1}.json')

        # Manually create a cache file
        test_response = "Hello! I'm doing well, thank you for asking."
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(test_response, f, ensure_ascii=False)

        # Test cache retrieval
        os.environ['API_KEY'] = 'test_key'  # Set a dummy API key
        try:
            response = tool.apply(messages=messages, model=model, max_completion_tokens=100)
            assert response == test_response, f"Expected '{test_response}', got '{response}'"
            print('✓ Cache retrieval working correctly')
        finally:
            if 'API_KEY' in os.environ:
                del os.environ['API_KEY']

        # Test 6: Convenience function
        print('Test 6: Convenience function...')
        try:
            # This should also read from cache
            os.environ['API_KEY'] = 'test_key'
            response = cached_requests(
                messages=messages, model=model, max_completion_tokens=100, cache_dir=test_cache_dir
            )
            assert response == test_response, 'Convenience function should return cached result'
            print('✓ Convenience function working correctly')
        finally:
            if 'API_KEY' in os.environ:
                del os.environ['API_KEY']

        print('\n🎉 All tests passed successfully!')

    except Exception as e:
        print(f'\n❌ Test failed with error: {str(e)}')
        raise
    finally:
        # Clean up test cache directory
        if os.path.exists(test_cache_dir):
            shutil.rmtree(test_cache_dir)
            print(f'✓ Cleaned up test cache directory: {test_cache_dir}')


if __name__ == '__main__':
    test_cached_requests_tool()
