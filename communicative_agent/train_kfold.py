import os
import pickle

from lib import utils
from lib.nn.data_scaler import DataScaler
from communicative_agent import CommunicativeAgent
import torch
from trainer import Trainer

NB_FOLDS = 5
DATASETS = [["pb2007"], ["gb2016", "th2016"]]
JERK_LOSS_WEIGHTS = [0, 0.15]


def train_agent(agent, save_path):
    print("Training %s" % save_path)
    if os.path.isdir(save_path):
        print("Already done")
        print()
        return
    device= "cuda" if torch.cuda.is_available() else "cpu",

    dataloaders = agent.get_dataloaders()
    optimizers = agent.get_optimizers()
    losses_fn = agent.get_losses_fn()

    sound_scalers = {
        "synthesizer": DataScaler.from_standard_scaler(
            agent.synthesizer.sound_scaler
        ).to(device),
        "agent": DataScaler.from_standard_scaler(agent.sound_scaler).to(device),
    }

    trainer = Trainer(
        agent.nn,
        optimizers,
        *dataloaders,
        losses_fn,
        agent.config["training"]["max_epochs"],
        agent.config["training"]["patience"],
        agent.synthesizer,
        sound_scalers,
        "./out/checkpoint.pt",
    )
    metrics_record = trainer.train()

    utils.mkdir(save_path)
    agent.save(save_path)
    with open(save_path + "/metrics.pickle", "wb") as f:
        pickle.dump(metrics_record, f)


def main():
    final_configs = utils.read_yaml_file("communicative_agent/communicative_final_configs.yaml")

    for datasets_name in DATASETS:
        datasets_key = ",".join(datasets_name)
        config = final_configs[datasets_key]

        for i_fold in range(NB_FOLDS):
            for jerk_loss_weight in JERK_LOSS_WEIGHTS:
                config["sound_quantizer"]["name"] = "kfold-%s-%s" % (datasets_key, i_fold)
                config["training"]["jerk_loss_weight"] = jerk_loss_weight

                agent = CommunicativeAgent(config)
                save_path = "out/communicative_agent/kfold-%s-jerk=%s-%s" % (datasets_key, jerk_loss_weight, i_fold)
                train_agent(agent, save_path)


if __name__ == "__main__":
    main()
