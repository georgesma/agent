import os
import pickle
import numpy as np
import random

from lib import utils
from quantizer import Quantizer
from trainer import Trainer
from lib.dataset_wrapper import Dataset

NB_FOLDS = 5
DATASETS = [["pb2007"], ["gb2016", "th2016"]]

def train_quantizer(quantizer, save_path):
    print("Training %s" % (save_path))
    if os.path.isdir(save_path):
        print("Already done")
        print()
        with open(save_path + "/metrics.pickle", "rb") as f:
            metrics_record = pickle.load(f)
        return metrics_record

    dataloaders = quantizer.get_dataloaders()
    optimizer = quantizer.get_optimizer()
    loss_fn = quantizer.get_loss_fn()

    trainer = Trainer(
        quantizer.nn,
        optimizer,
        *dataloaders,
        loss_fn,
        quantizer.config["training"]["max_epochs"],
        quantizer.config["training"]["patience"],
        "./out/checkpoint.pt",
    )
    metrics_record = trainer.train()

    utils.mkdir(save_path)
    quantizer.save(save_path)
    with open(save_path + "/metrics.pickle", "wb") as f:
        pickle.dump(metrics_record, f)

    return metrics_record


def main():
    final_configs = utils.read_yaml_file("quantizer/quantizer_final_configs.yaml")
    for datasets_name in DATASETS:
        datasets_key = ",".join(datasets_name)
        config = final_configs["%s-cepstrum" % datasets_key]

        datasets_folds = {}
        for dataset_name in datasets_name:
            dataset = Dataset(dataset_name)
            dataset_items = dataset.get_items_name("cepstrum")
            random.shuffle(dataset_items)
            dataset_folds = []
            nb_items = len(dataset_items)
            for i_fold in range(NB_FOLDS):
                fold_start = round(i_fold * nb_items / NB_FOLDS)
                fold_end = round((i_fold + 1) * nb_items / NB_FOLDS)
                dataset_folds.append(dataset_items[fold_start:fold_end])
            datasets_folds[dataset_name] = dataset_folds

        for i_fold in range(NB_FOLDS):
            quantizer = Quantizer(config)
            save_path = "out/quantizer/kfold-%s-%s" % (datasets_key, i_fold)

            fold_datasplits = {}
            for dataset_name, dataset_folds in datasets_folds.items():
                test_items = dataset_folds[i_fold]
                train_validation_items_folds = np.arange(NB_FOLDS - 1)
                train_validation_items_folds[i_fold:] += 1
                train_validation_items = [item_name for j_fold in train_validation_items_folds for item_name in dataset_folds[j_fold]]
                nb_train_validation_items = len(train_validation_items)
                nb_validation_items = round(nb_train_validation_items / 5)
                validation_items = train_validation_items[:nb_validation_items]
                train_items = train_validation_items[nb_validation_items:]
                dataset_split = [train_items, validation_items, test_items]
                fold_datasplits[dataset_name] = dataset_split

            quantizer.datasplits = fold_datasplits
            train_quantizer(quantizer, save_path)




if __name__ == "__main__":
    main()
