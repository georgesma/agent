import os
import pickle
import numpy as np

from lib import utils
from quantizer import Quantizer
from trainer import Trainer

NB_TRAINING = 5


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
    repeated_datasets = utils.read_yaml_file("quantizer/repeated_datasets.yaml")

    for config_name, config in final_configs.items():
        print(config_name)
        dataset_name, modalities = config_name.split("-")
        if not modalities.startswith("repeated"):
            continue

        for i_training in range(NB_TRAINING):
            repeated_name = repeated_datasets[dataset_name][modalities][i_training]
            modality_name = "agent_art_%s" % repeated_name
            config["dataset"]["data_types"] = [modality_name]

            config["dataset"]["datasplit_seed"] = i_training
            quantizer = Quantizer(config)
            signature = quantizer.get_signature()
            save_path = "out/quantizer/%s-%s" % (signature, i_training)

            train_quantizer(quantizer, save_path)


if __name__ == "__main__":
    main()
