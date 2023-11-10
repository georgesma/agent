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


def train_agent(agent, save_path):
    print("Training %s" % save_path)
    if os.path.isdir(save_path):
        print("Already done")
        print()
        return

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
    final_quantizer_configs = utils.read_yaml_file("quantizer/quantizer_final_configs.yaml")

    for config_name, config in final_configs.items():
        quantizer_name = config_name.split("-")[0]
        quantizer_config = final_quantizer_configs["%s-cepstrum" % quantizer_name]

        for i_training in range(NB_TRAINING):
            quantizer_config["dataset"]["datasplit_seed"] = i_training
            quantizer_signature = utils.get_variable_signature(quantizer_config)

            for jerk_loss_weight in JERK_LOSS_WEIGHTS:
                config["sound_quantizer"]["name"] = "%s-%s" % (quantizer_signature, i_training)
                config["training"]["jerk_loss_weight"] = jerk_loss_weight

                agent = CommunicativeAgent(config)
                signature = agent.get_signature()
                save_path = "out/communicative_agent/%s-%s" % (signature, i_training)
                train_agent(agent, save_path)


if __name__ == "__main__":
    main()
