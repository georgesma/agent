from lib.dataset_wrapper import Dataset
from lib import abx_utils

DATASET_NAME = "fsew0"
FEATURES_TYPE = "ema"
DISTANCE_METRIC = "euclidean"

dataset = Dataset(DATASET_NAME)
items_features = dataset.get_items_data(FEATURES_TYPE)

consonants_indexes = abx_utils.get_phones_indexes(
    dataset.lab,
    dataset.phones_infos["consonants"],
    dataset.phones_infos["vowels"],
)

groups_abx_score = abx_utils.get_groups_score(
    dataset.phones_infos["consonant_groups"], consonants_indexes, items_features, DISTANCE_METRIC, 10_000
)
print(DATASET_NAME, FEATURES_TYPE, DISTANCE_METRIC, groups_abx_score)
