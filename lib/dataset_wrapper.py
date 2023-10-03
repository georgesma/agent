import numpy as np
from lib import utils
from glob import glob
import os
from scipy.io import wavfile
import pickle


SILENCE_NAMES = ["__", "sil"]

INFOS_PATH = os.path.join(os.path.dirname(__file__), "..")
DATASETS_PATH = os.path.join(os.path.dirname(__file__), "../datasets")


class Dataset:
    def __init__(self, name):
        self.name = name
        self.path = "%s/%s" % (DATASETS_PATH, self.name)
        self.infos = utils.read_yaml_file("%s/datasets_infos.yaml" % INFOS_PATH)[
            self.name
        ]
        self.features_config = utils.read_yaml_file(
            "%s/features_config.yaml" % INFOS_PATH
        )
        self.phones_infos = utils.read_yaml_file("%s/phones_infos.yaml" % INFOS_PATH)[
            self.infos["phones_infos"] if "phones_infos" in self.infos else self.name
        ]

        if os.path.isfile("%s/art_model.pickle" % self.path):
            with open("%s/art_model.pickle" % self.path, "rb") as f:
                self.art_model = pickle.load(f)
            with open("%s/ema_limits.pickle" % self.path, "rb") as f:
                self.ema_limits = pickle.load(f)

        self.lab = self._get_items_lab()
        self.items_transcription = self._get_items_transcription()

        self.has_palate = os.path.isfile("%s/palate.bin" % self.path)
        if self.has_palate:
            self.palate = np.fromfile(
                "%s/palate.bin" % self.path, dtype="float32"
            ).reshape((-1, 2))

    def _get_items_lab(self):
        lab_pathname = "%s/lab/*.lab" % (self.path)
        items_path = glob(lab_pathname)
        items_path.sort()

        items_lab = {}
        for item_path in items_path:
            item_name = utils.parse_item_name(item_path)
            item_data = utils.read_lab_file(item_path)
            items_lab[item_name] = item_data

        return items_lab

    def _get_items_transcription(self):
        items_transcription = {}

        for item_name, item_lab in self.lab.items():
            item_phones = [
                self.phones_infos["notation"][label["name"]]
                if label["name"] in self.phones_infos["notation"]
                else label["name"]
                for label in item_lab
                if label["name"] not in SILENCE_NAMES
            ]
            item_transcription = "".join(item_phones)
            items_transcription[item_name] = item_transcription

        return items_transcription

    def get_modality_dim(self, modality):
        if modality == "cepstrum":
            return 18
        elif modality == "source":
            return 2
        elif modality == "ema":
            return len(self.infos["ema_coils_order"])
        elif modality == "art_params":
            return len(self.infos["ema_coils_order"]) // 2

    def get_items_name(self, modality):
        data_pathname = "%s/%s/*.bin" % (self.path, modality)
        items_path = glob(data_pathname)
        items_path.sort()
        items_name = [utils.parse_item_name(item_path) for item_path in items_path]
        return items_name

    def get_item_data(self, item_name, modalities, cut_silences=False):
        if type(modalities) is not list:
            modalities = [modalities]

        modalities_data = []
        for modality in modalities:
            item_path = "%s/%s/%s.bin" % (self.path, modality, item_name)

            modality_dim = self.get_modality_dim(modality)
            modality_data = np.fromfile(item_path, dtype="float32").reshape(
                (-1, modality_dim)
            )
            if cut_silences is True:
                modality_data = self.cut_item_silences(item_name, modality_data)
            modalities_data.append(modality_data)

        if len(modalities) == 1:
            item_data = modalities_data[0]
        else:
            item_data = np.concatenate(modalities_data, axis=1)

        return item_data

    def get_items_data(self, modalities, cut_silences=False):
        first_modality = modalities[0] if type(modalities) is list else modalities
        items_name = self.get_items_name(first_modality)

        items_data = {}
        for item_name in items_name:
            items_data[item_name] = self.get_item_data(
                item_name, modalities, cut_silences
            )

        return items_data

    def get_items_list(self, items_name=None):
        items_list = []
        if items_name is None:
            items_name = list(self.lab.keys())

        for item_name in items_name:
            items_list.append(
                ("%s %s" % (item_name, self.items_transcription[item_name]), item_name)
            )

        return items_list

    def cut_item_silences(self, item_name, item_data):
        item_lab = self.lab[item_name]
        assert (
            item_lab[0]["name"] in SILENCE_NAMES
            and item_lab[-1]["name"] in SILENCE_NAMES
        )
        start = item_lab[0]["end"]
        end = item_lab[-1]["start"]
        return item_data[start:end]

    def get_item_wave(self, item_name):
        wav_path = "%s/wav/%s.wav" % (self.path, item_name)
        sampling_rate, wave = wavfile.read(wav_path)
        return wave

    def art_to_ema(self, art_params):
        ema = art_params @ self.art_model["art_to_ema_w"] + self.art_model["ema_mean"]
        return ema
