from lib.dataset_wrapper import Dataset
from lib import abx_utils

DATASETS_NAME = ["fsew0", "msak0"]
DISTANCES = {
    "ema": {
        "metric": "euclidean",
        "weight": 1,
    },
    "cepstrum": {
        "metric": "euclidean",
        "weight": 2,
    },
}

def main():
    main_dataset = None
    datasets_lab = {}
    datasets_features = {}
    for dataset_name in DATASETS_NAME:
        dataset = Dataset(dataset_name)
        if main_dataset is None:
            main_dataset = dataset
        datasets_lab[dataset_name] = dataset.lab
        dataset_features = {}

        for feature_type in DISTANCES.keys():
            items_features = dataset.get_items_data(feature_type)
            dataset_features[feature_type] = items_features

        datasets_features[dataset_name] = dataset_features

    consonants = main_dataset.phones_infos["consonants"]
    vowels = main_dataset.phones_infos["vowels"]

    consonants_indexes = abx_utils.get_datasets_phones_indexes(
        datasets_lab,
        consonants,
        vowels,
    )

    abx_matrix = abx_utils.get_abx_matrix(consonants, consonants_indexes, datasets_features, DISTANCES, 100)
    print(abx_matrix)

if __name__ == "__main__":
    main()
