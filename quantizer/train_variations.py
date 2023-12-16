import os
import pickle
import numpy as np

from lib import utils
from quantizer import Quantizer
from trainer import Trainer

DATASET_NAME = "pb2007"
MODALITIES = ["cepstrum", "art_params"]
NB_TRAINING = 5
VARIATIONS = {
    "frame_padding": [0, 1, 2, 3, 5],
    "num_embeddings": [32, 64, 128, 256, 512, 1024, 2048],
}


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
    for modality in MODALITIES:
        for variation_parameter, variation_values in VARIATIONS.items():
            final_configs = utils.read_yaml_file(
                "quantizer/quantizer_final_configs.yaml"
            )
            config_name = "%s-%s" % (DATASET_NAME, modality)
            config = final_configs[config_name]

            for variation_value in variation_values:
                config["model"][variation_parameter] = variation_value

                for i_training in range(NB_TRAINING):
                    config["dataset"]["datasplit_seed"] = i_training
                    quantizer = Quantizer(config)
                    signature = quantizer.get_signature()
                    save_path = "out/quantizer/%s-%s" % (signature, i_training)

                    train_quantizer(quantizer, save_path)


if __name__ == "__main__":
    main()
