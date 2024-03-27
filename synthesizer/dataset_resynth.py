import numpy as np
from tqdm import tqdm
from scipy.io import wavfile

from lib.dataset_wrapper import Dataset
from lib import utils
from external import lpcynet
from synthesizer import Synthesizer

DATASET_NAME = "pb2007"
SYNTHESIZER_NAME = "custom_split_pb2007"

synthesizer_path = "out/synthesizer/%s" % SYNTHESIZER_NAME
synthesizer = Synthesizer.reload(synthesizer_path)

dataset = Dataset(DATASET_NAME)
items_art = dataset.get_items_data("art_params", cut_silences=True)
items_source = dataset.get_items_data("source", cut_silences=True)

export_folder = "datasets/%s/synthesizer_resynth" % DATASET_NAME
utils.mkdir(export_folder)

for item_name, item_art in tqdm(items_art.items()):
    item_cepstrum = synthesizer.synthesize(item_art)
    item_source = items_source[item_name]
    item_features = np.concatenate((item_cepstrum, item_source), axis=1)
    item_resynth = lpcynet.synthesize_frames(item_features)

    export_file = "%s/%s.wav" % (export_folder, item_name)
    wavfile.write(export_file, 16000, item_resynth)
