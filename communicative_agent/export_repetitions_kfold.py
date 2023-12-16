from tqdm import tqdm
from communicative_agent import CommunicativeAgent
from lib.dataset_wrapper import Dataset
from lib import utils
from scipy.io import wavfile
import numpy as np
from external import lpcynet

DATASETS = [
    ["pb2007"],
    ["gb2016", "th2016"],
]
NB_FOLDS = 5
JERK_LOSS_WEIGHTS = [
    0,
    0.15,
]

def export_agent_repetitions(agent, agent_name, datasets_name, datasplits):
    for dataset_name in datasets_name:
        print("%s repeats %s" % (agent_name, dataset_name))
        dataset = Dataset(dataset_name)
        sound_type = agent.sound_quantizer.config["dataset"]["data_types"]
        items_sound = dataset.get_items_data(sound_type)
        items_source = dataset.get_items_data("source")
        repetition_art_export_dir = "./datasets/%s/kfold_art_%s" % (dataset_name, agent_name)
        repetition_cepstrum_export_dir = "./datasets/%s/kfold_cepstrum_%s" % (dataset_name, agent_name)
        repetition_wav_export_dir = "./datasets/%s/kfold_wav_%s" % (dataset_name, agent_name)
        utils.mkdir(repetition_art_export_dir)
        utils.mkdir(repetition_cepstrum_export_dir)
        utils.mkdir(repetition_wav_export_dir)

        test_items = datasplits[dataset_name][2]

        for item_name in tqdm(test_items):
            item_sound = items_sound[item_name]
            repetition = agent.repeat(item_sound)
            repetition_art = repetition["art_estimated"]
            repetition_art_file_path = "%s/%s.bin" % (repetition_art_export_dir, item_name)
            repetition_art.tofile(repetition_art_file_path)

            repetition_cepstrum = repetition["sound_repeated"]
            repetition_cepstrum_file_path = "%s/%s.bin" % (repetition_cepstrum_export_dir, item_name)
            repetition_cepstrum.tofile(repetition_cepstrum_file_path)

            item_source = items_source[item_name]
            repetition_lpcnet_features = np.concatenate((repetition_cepstrum, item_source), axis=1)
            repetition_lpcnet_features = dataset.cut_item_silences(item_name, repetition_lpcnet_features)
            repetition_wav = lpcynet.synthesize_frames(repetition_lpcnet_features)
            repetition_wav_file_path = "%s/%s.wav" % (repetition_wav_export_dir, item_name)
            wavfile.write(repetition_wav_file_path, 16000, repetition_wav)


def main():
    for datasets_name in DATASETS:
        datasets_key = ",".join(datasets_name)

        for i_fold in range(NB_FOLDS):
            for jerk_loss_weight in JERK_LOSS_WEIGHTS:
                save_path = "out/communicative_agent/kfold-%s-jerk=%s-%s" % (datasets_key, jerk_loss_weight, i_fold)
                agent = CommunicativeAgent.reload(save_path)
                agent_datasplits = agent.sound_quantizer.datasplits
                agent_name = "%s-jerk=%s" % (datasets_key, jerk_loss_weight)
                export_agent_repetitions(agent, agent_name, datasets_name, agent_datasplits)

if __name__ == "__main__":
    main()
