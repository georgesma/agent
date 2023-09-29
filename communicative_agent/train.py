import os
import pickle

from lib import utils
from lib.nn.data_scaler import DataScaler
from communicative_agent import CommunicativeAgent

from trainer import Trainer

NB_TRAINING = 5
# DATASETS_NAME = ["pb2007", "msak0", "fsew0"]
DATASETS_NAME = ["pb2007"]
FRAME_PADDING = [2]
JERK_LOSS_WEIGHTS = [0, 0.15]
NB_DERIVATIVES = [0]
ART_TYPE = "art_params"


def main():
    for i_training in range(NB_TRAINING):
        for jerk_loss_weight in JERK_LOSS_WEIGHTS:
            agent_config = utils.read_yaml_file("communicative_agent/communicative_config.yaml")
            agent_config["training"]["jerk_loss_weight"] = jerk_loss_weight

            agent = CommunicativeAgent(agent_config)
            signature = agent.get_signature()
            save_path = "out/communicative_agent/%s-%s" % (signature, i_training)

            print("Training %s (i_training=%s)" % (signature, i_training))
            if os.path.isdir(save_path):
                print("Already done")
                print()
                continue

            dataloaders = agent.get_dataloaders()
            optimizers = agent.get_optimizers()
            losses_fn = agent.get_losses_fn()

            sound_scalers = {
                "synthesizer": DataScaler.from_standard_scaler(
                    agent.synthesizer.sound_scaler
                ).to("cuda"),
                "agent": DataScaler.from_standard_scaler(agent.sound_scaler).to("cuda"),
            }

            trainer = Trainer(
                agent.nn,
                optimizers,
                *dataloaders,
                losses_fn,
                agent_config["training"]["max_epochs"],
                agent_config["training"]["patience"],
                agent.synthesizer,
                sound_scalers,
                "./out/checkpoint.pt",
            )
            metrics_record = trainer.train()

            utils.mkdir(save_path)
            agent.save(save_path)
            with open(save_path + "/metrics.pickle", "wb") as f:
                pickle.dump(metrics_record, f)


if __name__ == "__main__":
    main()
