import random

from lib.dataset_wrapper import Dataset

TEST_ITEMS = {
    "pb2007": [
        "item_0000",
        "item_0001",
        "item_0002",
        "item_0003",
        "item_0004",
        "item_0005",
        "item_0006",
        "item_0007",
        "item_0010",
        "item_0331",
        "item_0332",
        "item_0333",
        "item_0334",
        "item_0335",
        "item_0338",
        "item_0339",
        "item_0340",
        "item_0341",
        "item_0342",
        "item_0392",
        "item_0393",
        "item_0394",
        "item_0395",
        "item_0396",
        "item_0397",
        "item_0398",
        "item_0399",
        "item_0400",
        "item_0401",
        "item_0427",
        "item_0428",
        "item_0429",
        "item_0433",
        "item_0434",
        "item_0435",
        "item_0436",
        "item_0437",
        "item_0438",
        "item_0439",
    ]
}
SPLITS_SIZE = [64, 16, 20]

datasplits = {}
for dataset_name, test_items in TEST_ITEMS.items():
    dataset = Dataset(dataset_name)
    dataset_items = dataset.get_items_name("cepstrum")
    nb_items = len(dataset_items)
    for test_item in test_items:
        dataset_items.remove(test_item)
    random.shuffle(dataset_items)

    train_set_len = round(nb_items / 100 * SPLITS_SIZE[0])
    validation_set_len = round(nb_items / 100 * SPLITS_SIZE[1])
    train_set = dataset_items[:train_set_len]
    dataset_items = dataset_items[train_set_len:]
    validation_set = dataset_items[:validation_set_len]
    dataset_items = dataset_items[validation_set_len:]
    test_set = [*test_items, *dataset_items]
    datasplits[dataset_name] = [train_set, validation_set, test_set]

# quantizer.datasplits = datasplits
