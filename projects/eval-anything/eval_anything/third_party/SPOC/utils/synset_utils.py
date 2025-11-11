# Copyright 2024 Allen Institute for AI
# ==============================================================================

from functools import lru_cache
from typing import List, Set, Union

from nltk.corpus import wordnet2022 as wn
from nltk.corpus.reader import Synset

from eval_anything.third_party.SPOC.utils.constants.object_constants import (
    AI2THOR_OBJECT_TYPE_TO_WORDNET_SYNSET,
)
from eval_anything.third_party.SPOC.utils.objaverse_annotation import get_objaverse_annotations


def generate_all_hypernyms_with_exclusions(
    synset: Union[str, Synset],
    excluded: Union[Set[str], str],
    include_self_synset: bool = True,
) -> Set[Synset]:
    if isinstance(synset, str):
        synset = wn.synset(synset)

    return {
        h
        for hp in synset.hypernym_paths()
        for h in hp
        if (include_self_synset or h != synset) and h.name() not in excluded
    }


@lru_cache(maxsize=10000, typed=True)
def is_hypernym_of(synset: Union[str, Synset], possible_hypernym: Union[str, Synset]) -> bool:
    if isinstance(synset, str):
        synset = wn.synset(synset)

    if isinstance(possible_hypernym, str):
        possible_hypernym = wn.synset(possible_hypernym)

    return possible_hypernym in synset.lowest_common_hypernyms(possible_hypernym)


def get_all_synsets_in_spoc() -> List[Synset]:
    synsets = {ann['synset'] for ann in get_objaverse_annotations().values()} | set(
        AI2THOR_OBJECT_TYPE_TO_WORDNET_SYNSET.values()
    )
    synsets = sorted(list({wn.synset(s) for s in synsets}), key=lambda s: s.name())
    return synsets
