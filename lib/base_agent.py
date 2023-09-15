from lib import utils
from lib.dataset_wrapper import Dataset


class BaseAgent:
    def get_signature(self):
        return utils.get_variable_signature(self.config)

    def get_main_dataset(self):
        return Dataset(self.config["dataset"]["names"][0])

    def repeat_datasplit(self, datasplit_index=None):
        agent_features = {}
        sound_type = self.config["dataset"]["sound_type"]

        for dataset_name in self.config["dataset"]["names"]:
            dataset_features = {}

            dataset = Dataset(dataset_name)
            if datasplit_index is None:
                items_name = dataset.get_items_name(sound_type)
            else:
                items_name = self.datasplits[dataset_name][datasplit_index]

            items_sound = dataset.get_items_data(self.config["dataset"]["sound_type"])
            for item_name in items_name:
                item_sound = items_sound[item_name]
                repetition = self.repeat(item_sound)
                for repetition_type, repetition_data in repetition.items():
                    if repetition_type not in dataset_features:
                        dataset_features[repetition_type] = {}
                    dataset_features[repetition_type][item_name] = repetition_data

            agent_features[dataset_name] = dataset_features
        return agent_features

    def get_datasplit_lab(self, datasplit_index=None):
        datasplit_lab = {}

        for dataset_name in self.config["dataset"]["names"]:
            dataset = Dataset(dataset_name)

            if datasplit_index is None:
                dataset_lab = dataset.lab
            else:
                dataset_split = self.datasplits[dataset_name][datasplit_index]
                dataset_lab = {
                    item_name: dataset.lab[item_name] for item_name in dataset_split
                }

            datasplit_lab[dataset_name] = dataset_lab

        return datasplit_lab
