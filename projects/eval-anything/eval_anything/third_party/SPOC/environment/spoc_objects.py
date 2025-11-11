# Copyright 2024 Allen Institute for AI
# ==============================================================================

import json
from typing import Any, Dict

from torch.distributions.utils import lazy_property

from eval_anything.third_party.SPOC.utils.constants.object_constants import (
    AI2THOR_OBJECT_TYPE_TO_MOST_SPECIFIC_WORDNET_LEMMA,
    AI2THOR_OBJECT_TYPE_TO_WORDNET_SYNSET,
)
from eval_anything.third_party.SPOC.utils.objaverse_annotation import get_objaverse_annotations


class SPOCObject(dict):
    _ALWAYS_KEYS = {'isObjaverse', 'synset', 'lemma'}

    def __init__(self, thor_obj: Dict[str, Any]):
        super().__init__()
        self._thor_obj = thor_obj
        self._cache = {}

    @lazy_property
    def is_objaverse(self):
        return self._thor_obj['assetId'] in get_objaverse_annotations()

    @lazy_property
    def annotation(self):
        if self.is_objaverse:
            return get_objaverse_annotations()[self._thor_obj['assetId']]
        return {}

    def __getitem__(self, item):
        if self.is_objaverse and item == 'objectType' and self._thor_obj[item] == 'Undefined':
            return self._thor_obj['objectId'].split('|')[0]

        if item in self._thor_obj:
            return self._thor_obj[item]

        if item in self._cache:
            return self._cache[item]

        if item == 'isObjaverse':
            return self.is_objaverse

        if item == 'synset':
            if self.is_objaverse:
                self._cache[item] = self.annotation['synset']
            else:
                self._cache[item] = AI2THOR_OBJECT_TYPE_TO_WORDNET_SYNSET[
                    self._thor_obj['objectType']
                ]

        elif item == 'lemma':
            if self.is_objaverse:
                self._cache[item] = self.annotation['most_specific_lemma']
            else:
                self._cache[item] = AI2THOR_OBJECT_TYPE_TO_MOST_SPECIFIC_WORDNET_LEMMA[
                    self._thor_obj['objectType']
                ]

        elif item in self.annotation:
            self._cache[item] = self.annotation[item]

        elif not self.is_objaverse and item == 'description':
            self._cache[item] = f"undescribed THOR item, type {self._thor_obj['objectType']}"

        else:
            raise ValueError(f'Unknown key {item}')

        return self._cache[item]

    def __setitem__(self, key, value):
        if key in self._thor_obj:
            self._thor_obj[key] = value
        else:
            self._cache[key] = value

    def _key_set(self):
        keys = set(self._thor_obj.keys())
        keys.update(self._cache.keys())
        keys.update(self._ALWAYS_KEYS)
        keys.update(self.annotation.keys())
        return keys

    def keys(self):
        return iter(self._key_set())

    def values(self):
        return map(self.__getitem__, self.keys())

    def __iter__(self):
        yield from self.keys()

    def items(self):
        for key in self.keys():
            yield key, self[key]

    def __contains__(self, key):
        return (
            key in self._thor_obj
            or key in self._cache
            or key in self._ALWAYS_KEYS
            or key in self.annotation
        )

    def __eq__(self, other):
        if not isinstance(other, SPOCObject):
            return False
        self_keys = self._key_set()
        other_keys = other._key_set()
        return (self_keys == other_keys) and all(self[key] == other[key] for key in self_keys)

    def __str__(self):
        return json.dumps(
            {**self._thor_obj, **self._cache, 'isObjaverse': self.is_objaverse}, indent=2
        )

    # noinspection PyStatementEffect
    def __repr__(self):
        return (
            f'SPOCObject(dict('
            f"objectId='{self['objectId']}',"
            f" objectType='{self['objectType']}',"
            f" assetId='{self['assetId']}',"
            f" isObjaverse={self['isObjaverse']},"
            f'))'
        )
