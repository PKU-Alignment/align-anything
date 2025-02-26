import prior

SYNSET_TO_BEST_LEMMA = prior.load_dataset(
    dataset="spoc-data",
    entity="spoc-robot",
    revision="objaverse-annotation-plus",
    which_dataset="synset_to_best_lemma",
)["train"].data

SYNSET_TO_BEST_LEMMA = prior.load_dataset(
    dataset="spoc-data",
    entity="spoc-robot",
    revision="objaverse-annotation-plus",
    which_dataset="synset_to_best_lemma",
)["train"].data
