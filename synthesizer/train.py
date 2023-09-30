import os
import pickle

from lib import utils
from synthesizer import Synthesizer
from trainer import Trainer

NB_TRAINING = 5
DATASETS_NAME = ["pb2007"]


def main():
    for i_training in range(NB_TRAINING):
        for dataset_name in DATASETS_NAME:
            synthesizer_config = utils.read_yaml_file(
                "synthesizer/synthesizer_config.yaml"
            )
            synthesizer_config["dataset"]["name"] = dataset_name

            synthesizer = Synthesizer(synthesizer_config)
            signature = synthesizer.get_signature()
            save_path = "out/synthesizer/%s-%s" % (signature, i_training)

            print("Training %s (i_training=%s)" % (signature, i_training))
            if os.path.isdir(save_path):
                print("Already done")
                print()
                continue

            dataloaders = synthesizer.get_dataloaders()
            optimizer = synthesizer.get_optimizer()
            loss_fn = synthesizer.get_loss_fn()

            trainer = Trainer(
                synthesizer.nn,
                optimizer,
                *dataloaders,
                loss_fn,
                synthesizer_config["training"]["max_epochs"],
                synthesizer_config["training"]["patience"],
                "./out/checkpoint.pt",
            )
            metrics_record = trainer.train()

            utils.mkdir(save_path)
            synthesizer.save(save_path)
            with open(save_path + "/metrics.pickle", "wb") as f:
                pickle.dump(metrics_record, f)


if __name__ == "__main__":
    main()
