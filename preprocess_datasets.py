from glob import glob
import librosa
import numpy as np
from tqdm import tqdm
from scipy.io import wavfile
import pickle
import math

from lib import utils
from lib import art_model
from external import lpcynet

INT16_MAX_VALUE = 32767


def rms(y):
    return np.sqrt(np.mean((y * 1.0) ** 2))


def compute_wav_rms(wav_pathname):
    wavfiles_path = glob(wav_pathname)

    dataset_pcm = []
    for wavfile_path in tqdm(wavfiles_path):
        pcm, _ = librosa.load(wavfile_path, sr=None)
        dataset_pcm.append(pcm)

    dataset_pcm = np.concatenate(dataset_pcm, axis=0)
    dataset_wav_rms = rms(dataset_pcm)
    return dataset_wav_rms


def preprocess_wav(
    dataset_name, wav_pathname, target_sampling_rate, dataset_wav_rms, target_wav_rms
):
    export_dir = "datasets/%s/wav" % dataset_name
    utils.mkdir(export_dir)

    wav_scaling_factor = 1
    if target_wav_rms is not None:
        wav_scaling_factor = np.sqrt(target_wav_rms ** 2 / dataset_wav_rms ** 2)

    wavfiles_path = glob(wav_pathname)

    for wavfile_path in tqdm(wavfiles_path):
        pcm, wavfile_sampling_rate = librosa.load(wavfile_path, sr=None)

        pcm = pcm * wav_scaling_factor
        if wavfile_sampling_rate != target_sampling_rate:
            pcm = librosa.resample(pcm, wavfile_sampling_rate, target_sampling_rate)

        pcm = pcm * INT16_MAX_VALUE
        assert np.abs(pcm).max() <= INT16_MAX_VALUE
        pcm = pcm.astype("int16")
        item_name = utils.parse_item_name(wavfile_path)
        wavfile.write("%s/%s.wav" % (export_dir, item_name), target_sampling_rate, pcm)


def extract_cepstrum_and_source(dataset_name):
    cepstrum_export_dir = "datasets/%s/cepstrum" % dataset_name
    utils.mkdir(cepstrum_export_dir)
    source_export_dir = "datasets/%s/source" % dataset_name
    utils.mkdir(source_export_dir)

    wavfiles_dir = "datasets/%s/wav" % dataset_name
    wavfiles_path = glob("%s/*.wav" % wavfiles_dir)

    for wavfile_path in tqdm(wavfiles_path):
        item_name = utils.parse_item_name(wavfile_path)

        sr, pcm = wavfile.read(wavfile_path)
        assert sr == 16000
        item_lpcnet_features = lpcynet.analyze_frames(pcm)

        item_cepstrum = item_lpcnet_features[:, :18]
        item_source = item_lpcnet_features[:, 18:]

        item_cepstrum.tofile("%s/%s.bin" % (cepstrum_export_dir, item_name))
        item_source.tofile("%s/%s.bin" % (source_export_dir, item_name))


def preprocess_ema(
    dataset_name,
    ema_pathname,
    ema_format,
    ema_sampling_rate,
    ema_scaling_factor,
    ema_coils_order,
    ema_needs_lowpass,
    target_sampling_rate,
):
    export_dir = "datasets/%s/ema" % dataset_name
    utils.mkdir(export_dir)

    items_ema = {}
    ema_limits = {
        "xmin": math.nan,
        "xmax": math.nan,
        "ymin": math.nan,
        "ymax": math.nan,
    }

    if ema_needs_lowpass:
        lowpass_filter = utils.create_lowpass_filter(ema_sampling_rate, 50)

    emafiles_path = glob(ema_pathname)
    for emafile_path in tqdm(emafiles_path):
        item_ema = utils.read_ema_file(emafile_path, ema_format)

        # lowpass filtering
        if ema_needs_lowpass:
            item_ema = lowpass_filter(item_ema)

        # resampling
        if ema_sampling_rate != target_sampling_rate:
            item_ema = utils.interp_2d(
                item_ema, ema_sampling_rate, target_sampling_rate
            )

        # reordering, target coils order:
        #   lower incisor, tongue tip, tongue middle, tongue back, lower lip, upper lip and velum
        item_ema = item_ema[:, ema_coils_order]

        # scaling to mm
        item_ema = item_ema / ema_scaling_factor

        item_name = utils.parse_item_name(emafile_path)
        item_ema.astype("float32").tofile("%s/%s.bin" % (export_dir, item_name))
        items_ema[item_name] = item_ema

        ema_limits["xmin"] = min(item_ema[:, 0::2].min(), ema_limits["xmin"])
        ema_limits["xmax"] = max(item_ema[:, 0::2].max(), ema_limits["xmax"])
        ema_limits["ymin"] = min(item_ema[:, 1::2].min(), ema_limits["ymin"])
        ema_limits["ymax"] = max(item_ema[:, 1::2].max(), ema_limits["ymax"])

    with open("datasets/%s/ema_limits.pickle" % dataset_name, "wb") as file:
        pickle.dump(ema_limits, file)

    return items_ema


def extract_art_parameters(dataset_name, items_ema):
    dataset_dir = "datasets/%s" % dataset_name

    all_ema_frames = np.concatenate(list(items_ema.values()), axis=0)
    art_model_params = art_model.build_art_model(all_ema_frames)
    with open("%s/art_model.pickle" % dataset_dir, "wb") as file:
        pickle.dump(art_model_params, file)

    export_dir = "datasets/%s/art_params" % dataset_name
    utils.mkdir(export_dir)

    for item_name, item_ema in tqdm(items_ema.items()):
        item_art = art_model.ema_to_art(art_model_params, item_ema)
        item_art.astype("float32").tofile("%s/%s.bin" % (export_dir, item_name))


def preprocess_lab(dataset_name, lab_pathname, dataset_resolution, target_resolution):
    export_dir = "datasets/%s/lab" % dataset_name
    utils.mkdir(export_dir)

    resolution_ratio = target_resolution / dataset_resolution

    labfiles_path = glob(lab_pathname)
    for labfile_path in tqdm(labfiles_path):
        item_lab = utils.read_lab_file(labfile_path, resolution_ratio)
        item_name = utils.parse_item_name(labfile_path)
        utils.save_lab_file("%s/%s.lab" % (export_dir, item_name), item_lab)


def main():
    features_config = utils.read_yaml_file("./features_config.yaml")
    datasets_infos = utils.read_yaml_file("./datasets_infos.yaml")
    datasets_wav_rms = {}

    for dataset_name, dataset_infos in datasets_infos.items():
        print("Preprocessing %s..." % dataset_name)

        print("Computing RMS...")
        dataset_wav_rms = compute_wav_rms(dataset_infos["wav_pathname"])
        datasets_wav_rms[dataset_name] = dataset_wav_rms
        print("Computing RMS done")

        print("Resampling WAV files...")
        target_wav_rms = (
            datasets_wav_rms[dataset_infos["wav_rms_reference"]]
            if "wav_rms_reference" in dataset_infos
            else None
        )
        preprocess_wav(
            dataset_name,
            dataset_infos["wav_pathname"],
            features_config["wav_sampling_rate"],
            dataset_wav_rms,
            target_wav_rms,
        )
        print("Resampling WAV files done")

        print("Extracting cepstrograms and source parameters...")
        extract_cepstrum_and_source(dataset_name)
        print("Extracting cepstrograms and source parameters done")

        if "ema_pathname" in dataset_infos:
            frames_sampling_rate = features_config["ema_sampling_rate"]

            print("Preprocessing EMA...")
            items_ema = preprocess_ema(
                dataset_name,
                dataset_infos["ema_pathname"],
                dataset_infos["ema_format"],
                dataset_infos["ema_sampling_rate"],
                dataset_infos["ema_scaling_factor"],
                dataset_infos["ema_coils_order"],
                dataset_infos["ema_needs_lowpass"],
                frames_sampling_rate,
            )
            print("Preprocessing EMA done")

            print("Extracting articulatory model and parameters...")
            extract_art_parameters(dataset_name, items_ema)
            print("Extracting articulatory model and parameters done")

        print("Resampling LAB files...")
        preprocess_lab(
            dataset_name,
            dataset_infos["lab_pathname"],
            dataset_infos["lab_resolution"],
            frames_sampling_rate,
        )
        print("Resampling LAB files done")

        print("Preprocessing %s done" % dataset_name)
        print("")

        # TODO: add the palate importation
        # it should be placed in a palate.bin file placed at the root of the ./datasets/DATASET_NAME folder
        # it should be a float32 numpy array of shape (point_number, 2) saved with array.tofile


if __name__ == "__main__":
    main()
