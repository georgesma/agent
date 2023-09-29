import os
import pickle
from hyperopt import tpe, hp, fmin
import numpy as np

from lib import utils
from lib import abx_utils
from quantizer import Quantizer
from trainer import Trainer

DATASPLIT_SEED = 1337
ABX_NB_SAMPLES = 50
QUANTIZER_ABX_DISTANCE = {"quantized_latent": {"metric": "cosine", "weight": 1}}


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


def get_quantizer_abx_score(quantizer):
    main_dataset = quantizer.main_dataset
    quantizer_lab = quantizer.get_datasplit_lab(2)
    quantizer_features = quantizer.autoencode_datasplit(2)

    consonants = main_dataset.phones_infos["consonants"]
    vowels = main_dataset.phones_infos["vowels"]
    consonants_indexes = abx_utils.get_datasets_phones_indexes(
        quantizer_lab, consonants, vowels
    )

    abx_matrix = abx_utils.get_abx_matrix(
        consonants,
        consonants_indexes,
        quantizer_features,
        QUANTIZER_ABX_DISTANCE,
        ABX_NB_SAMPLES,
    )
    global_abx_score = abx_utils.get_global_score(abx_matrix)
    return global_abx_score


def train_with_hyperparameters(hyperparameters):
    quantizer_config = utils.read_yaml_file("quantizer/quantizer_config.yaml")
    quantizer_config["dataset"]["datasplit_seed"] = DATASPLIT_SEED

    quantizer_config["training"]["learning_rate"] = hyperparameters["learning_rate"]
    quantizer_config["model"]["dropout_p"] = hyperparameters["dropout_p"]
    quantizer_config["model"]["commitment_cost"] = hyperparameters["commitment_cost"]
    quantizer_config["model"]["hidden_dims"] = [
        int(2 ** hyperparameters["dim_hidden_layers"])
    ] * int(hyperparameters["nb_hidden_layers"])
    quantizer_config["model"]["num_embeddings"] = int(2 ** hyperparameters["num_embeddings"])
    quantizer_config["model"]["embedding_dim"] = int(2 ** hyperparameters["embedding_dim"])

    quantizer = Quantizer(quantizer_config)
    signature = quantizer.get_signature()
    save_path = "out/quantizer/%s" % (signature)

    train_quantizer(quantizer, save_path)
    abx_score = get_quantizer_abx_score(quantizer)
    return 100 - abx_score


def main():
    hyperparameters_space = {
        "learning_rate": hp.loguniform("learning_rate", np.log(1e-4), np.log(1e-2)),
        "dropout_p": hp.uniform("dropout_p", 0, 0.9),
        "dim_hidden_layers": hp.quniform("dim_hidden_layers", 6, 9, 1),
        "nb_hidden_layers": hp.quniform("nb_hidden_layers", 1, 4, 1),
        "commitment_cost": hp.uniform("commitment_cost", 0.1, 2),
        "num_embeddings": hp.quniform("num_embeddings", 5, 9, 1),
        "embedding_dim": hp.quniform("embedding_dim", 3, 7, 1),
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
