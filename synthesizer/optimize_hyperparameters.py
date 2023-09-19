import os
import pickle
from hyperopt import tpe, hp, fmin
import numpy as np

from lib import utils
from synthesizer import Synthesizer
from trainer import Trainer

DATASPLIT_SEED = 1337


def train_synthesizer(synthesizer, save_path):
    print("Training %s" % (save_path))
    if os.path.isdir(save_path):
        print("Already done")
        print()
        with open(save_path + "/metrics.pickle", "rb") as f:
            metrics_record = pickle.load(f)
        return metrics_record

    dataloaders = synthesizer.get_dataloaders()
    optimizer = synthesizer.get_optimizer()
    loss_fn = synthesizer.get_loss_fn()

    trainer = Trainer(
        synthesizer.nn,
        optimizer,
        *dataloaders,
        loss_fn,
        synthesizer.config["training"]["max_epochs"],
        synthesizer.config["training"]["patience"],
        "./out/checkpoint.pt",
    )
    metrics_record = trainer.train()

    utils.mkdir(save_path)
    synthesizer.save(save_path)
    with open(save_path + "/metrics.pickle", "wb") as f:
        pickle.dump(metrics_record, f)

    return metrics_record


def train_with_hyperparameters(hyperparameters):
    synthesizer_config = utils.read_yaml_file("synthesizer/synthesizer_config.yaml")
    synthesizer_config["dataset"]["datasplit_seed"] = DATASPLIT_SEED
    synthesizer_config["training"]["learning_rate"] = hyperparameters["learning_rate"]
    synthesizer_config["model"]["dropout_p"] = hyperparameters["dropout_p"]
    synthesizer_config["model"]["hidden_layers"] = [
        int(2 ** hyperparameters["dim_hidden_layers"])
    ] * int(hyperparameters["nb_hidden_layers"])

    synthesizer = Synthesizer(synthesizer_config)
    signature = synthesizer.get_signature()
    save_path = "out/synthesizer/%s" % (signature)

    metrics_record = train_synthesizer(synthesizer, save_path)
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
