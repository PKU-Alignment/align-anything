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
Cache management utilities for storing and retrieving inference results.

Use abstract cache keys instead of concrete file paths for `save` and `load` interfaces,
so that the cache manager is independent of the file system. This is convenient for
switching to other storage backends in the future.

Use pickle to automatically handle the serialization and deserialization of complex objects.
"""

import os
import pickle
from collections import namedtuple
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from eval_anything.utils.data_type import InferenceInput, InferenceOutput
from eval_anything.utils.logger import EvalLogger
from eval_anything.utils.uuid import UUIDGenerator


class BinaryCache:
    """Binary cache implementation with support for multiple tensor types"""

    def __init__(self, cache_dir: str = '.cache/eval_anything', logger: EvalLogger = None):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger
        self.uuid_generator = UUIDGenerator()

    def _get_cache_path(self, key: Dict[str, Any]) -> Path:
        """Get cache file path from key"""
        return self.cache_dir / f'{self.uuid_generator(key)}.pkl'

    def is_cached(self, key: Dict[str, Any]) -> bool:
        """Check if the key is cached"""
        cache_path = self._get_cache_path(key)
        return cache_path.exists()

    def get(self, key: Dict[str, Any]) -> Optional[Any]:
        """Retrieve object from cache"""
        cache_path = self._get_cache_path(key)
        if not cache_path.exists():
            # self.logger.log('info', f"Cache miss for key: {key}")
            return None
        try:
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
            self.logger.log(
                'info', f'Get inference outputs from cache: {os.path.join(__file__, cache_path)}'
            )
            return data
        except Exception as e:
            self.logger.log('error', f'Failed to load cached object with key {key}. Error: {e}')
            return None

    def put(self, key: Dict[str, Any], value: Any) -> bool:
        """Store object in cache"""
        cache_path = self._get_cache_path(key)
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(value, f)
            self.logger.log(
                'info', f'Save inference outputs to cache: {os.path.join(__file__, cache_path)}'
            )
            return True
        except Exception as e:
            self.logger.log('error', f'Failed to cache object with key {key}. Error: {e}')
            return False

    def clear(self):
        """Clear all cache files"""
        count = 0
        for cache_file in self.cache_dir.glob('*.pkl'):
            try:
                cache_file.unlink()
                count += 1
            except Exception as e:
                self.logger.log(
                    'error',
                    f'Failed to delete cache file {os.path.join(__file__, cache_file)}. Error: {e}',
                )
        self.logger.log('info', f'Cleared {count} cache files from {self.cache_dir}')


class CacheManager:
    """Cache manager for inference results"""

    def __init__(self, cache_dir: str, logger: EvalLogger):
        """Initialize cache manager

        Args:
            cache_dir: Directory path for caching results
        """
        self.cache_dir = cache_dir
        self.binary_cache = BinaryCache(cache_dir, logger)
        self.uuid_generator = UUIDGenerator()

    def get_cache_path(
        self, model_cfg: namedtuple, infer_cfg: namedtuple, inputs: List[InferenceInput]
    ) -> Tuple[str, bool]:
        cache_key = self.uuid_generator(
            {'model_cfg': model_cfg, 'infer_cfg': infer_cfg, 'inputs': inputs}
        )
        return cache_key, self.binary_cache.is_cached(cache_key)

    def _normalize_value(self, value: any) -> str:
        """Normalize values for consistent hashing"""
        if isinstance(value, float):
            return f'{value:.6f}'
        return str(value)

    def save(self, cache_key: str, outputs: List[InferenceOutput]) -> None:
        """Save outputs to cache"""
        self.binary_cache.put(cache_key, outputs)

    def load(self, cache_key: str) -> List[InferenceOutput]:
        """Load outputs from cache"""
        return self.binary_cache.get(cache_key)

    def clear(self) -> None:
        """Clear all cached inference results"""
        self.binary_cache.clear()
