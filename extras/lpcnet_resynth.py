import numpy as np
from tqdm import tqdm
from scipy.io import wavfile

from lib.dataset_wrapper import Dataset
from lib import utils
from external import lpcynet

DATASET_NAME = "pb2007"
ITEMS_NAME = [
    "item_0000",
    "item_0001",
    "item_0002",
    "item_0003",
    "item_0004",
    "item_0005",
    "item_0006",
    "item_0007",
    "item_0010",
    "item_0331",
    "item_0332",
    "item_0333",
    "item_0334",
    "item_0335",
    "item_0338",
    "item_0339",
    "item_0340",
    "item_0341",
    "item_0342",
    "item_0392",
    "item_0393",
    "item_0394",
    "item_0395",
    "item_0396",
    "item_0397",
    "item_0398",
    "item_0399",
    "item_0400",
    "item_0401",
    "item_0427",
    "item_0428",
    "item_0429",
    "item_0433",
    "item_0434",
    "item_0435",
    "item_0436",
    "item_0437",
    "item_0438",
    "item_0439",
]

dataset = Dataset(DATASET_NAME)
items_cepstrum = dataset.get_items_data("cepstrum", cut_silences=True)
items_source = dataset.get_items_data("source", cut_silences=True)

export_folder = "datasets/%s/lpcnet_resynth" % DATASET_NAME
utils.mkdir(export_folder)

for item_name in tqdm(ITEMS_NAME):
    item_cepstrum = items_cepstrum[item_name]
    item_source = items_source[item_name]
    item_features = np.concatenate((item_cepstrum, item_source), axis=1)
    item_resynth = lpcynet.synthesize_frames(item_features)

    export_file = "%s/%s.wav" % (export_folder, item_name)
    wavfile.write(export_file, 16000, item_resynth)
