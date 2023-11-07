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

for agent_name in AGENTS:
    agent_path = "./out/communicative_agent/%s" % agent_name
    agent = CommunicativeAgent.reload(agent_path)

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
