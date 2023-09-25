import os
import pickle
import numpy as np

from lib import utils
from quantizer import Quantizer
from trainer import Trainer


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
    quantizer_config = utils.read_yaml_file("quantizer/quantizer_config.yaml")
    quantizer = Quantizer(quantizer_config)
    signature = quantizer.get_signature()
    save_path = "out/quantizer/%s" % (signature)

    train_quantizer(quantizer, save_path)


if __name__ == "__main__":
    main()
