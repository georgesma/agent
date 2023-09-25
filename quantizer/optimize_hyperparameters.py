import os
import pickle
from hyperopt import tpe, hp, fmin
import numpy as np

from lib import utils
from quantizer import Quantizer
from trainer import Trainer

DATASPLIT_SEED = 1337


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


def train_with_hyperparameters(hyperparameters):
    quantizer_config = utils.read_yaml_file("quantizer/quantizer_config.yaml")
    # quantizer_config["dataset"]["datasplit_seed"] = DATASPLIT_SEED
    # quantizer_config["training"]["learning_rate"] = hyperparameters["learning_rate"]
    # quantizer_config["model"]["dropout_p"] = hyperparameters["dropout_p"]
    # quantizer_config["model"]["hidden_layers"] = [
    #     int(2 ** hyperparameters["dim_hidden_layers"])
    # ] * int(hyperparameters["nb_hidden_layers"])

    quantizer = Quantizer(quantizer_config)
    signature = quantizer.get_signature()
    save_path = "out/quantizer/%s" % (signature)

    metrics_record = train_quantizer(quantizer, save_path)
    final_validation_loss = min(metrics_record["validation"]["total"])
    return final_validation_loss


def main():
    hyperparameters_space = {
        "learning_rate": hp.loguniform("learning_rate", np.log(1e-4), np.log(1e-2)),
        "dropout_p": hp.uniform("dropout_p", 0, 0.9),
        "dim_hidden_layers": hp.quniform("dim_hidden_layers", 6, 9, 1),
        "nb_hidden_layers": hp.quniform("nb_hidden_layers", 1, 4, 1),
    }

    best_config = fmin(
        fn=train_with_hyperparameters,
        space=hyperparameters_space,
        algo=tpe.suggest,
        max_evals=100,
    )
    print("best config:")
    print(best_config)


if __name__ == "__main__":
    main()
