from tqdm import tqdm
from communicative_agent import CommunicativeAgent
from lib.dataset_wrapper import Dataset
from lib import utils

AGENTS = [
    "97a400f946a1202ec39bbf5546749656-2", # jerk_loss_weight = 0
    "aabf90478c0629fd266913e4b0ea1b72-3", # jerk_loss_weight = 0.15
]
DATASETS = [
    "pb2007",
]
NB_TRAINING = 5
JERK_LOSS_WEIGHTS = [0, 0.15]

def export_agent_art(agent, agent_name):
    for dataset_name in DATASETS:
        print("%s repeats %s" % (agent_name, dataset_name))
        dataset = Dataset(dataset_name)
        sound_type = agent.sound_quantizer.config["dataset"]["data_types"]
        items_sound = dataset.get_items_data(sound_type)
        repetition_export_dir = "./datasets/%s/agent_art_%s" % (dataset_name, agent_name)
        utils.mkdir(repetition_export_dir)

        for item_name, item_sound in tqdm(items_sound.items()):
            repetition = agent.repeat(item_sound)
            repetition_art = repetition["art_estimated"]
            repetition_file_path = "%s/%s.bin" % (repetition_export_dir, item_name)
            repetition_art.tofile(repetition_file_path)


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

                agent_signature = utils.get_variable_signature(config)

                # is_synth_agent = "use_synth_as_direct_model" in config["model"]
                # key = "repeated"
                # if is_synth_agent:
                #     key += "_synthasdirect"
                # if jerk_loss_weight > 0:
                #     key += "_jerkloss"
                # print(key, "%s-%s" % (agent_signature, i_training))
                # continue

                agent_name = "%s-%s" % (agent_signature, i_training)
                agent_path = "out/communicative_agent/%s" % agent_name
                agent = CommunicativeAgent.reload(agent_path)
                export_agent_art(agent, agent_name)


if __name__ == "__main__":
    main()
