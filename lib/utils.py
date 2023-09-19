import pathlib
import numpy as np
import yaml
import re
import random
import hashlib
from scipy import signal
import pickle
import os.path

RE_ITEM_NAME = re.compile(r"([\w-]+)\.[\w]+$")
RE_SPACES = re.compile(r"\s+")
RE_LAB_LINE = re.compile(r"^(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s+(.+)\n$")


def mkdir(path):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)


def read_yaml_file(filepath):
    with open(filepath, "r") as file:
        return yaml.safe_load(file)


def pickle_load(filepath, default_value):
    if not os.path.isfile(filepath):
        return default_value
    else:
        with open(filepath, "rb") as f:
            return pickle.load(f)


def pickle_dump(filepath, value):
    with open(filepath, "wb") as f:
        pickle.dump(value, f)


def read_est_file(filepath):
    with open(filepath, "rb") as f:
        header_end = b"EST_Header_End\n"
        header_end_len = len(header_end)
        header_end_pos = f.read().find(b"EST_Header_End\n")

        f.seek(header_end_pos + header_end_len)
        return np.fromfile(f, dtype="float32").reshape((-1, 22))[:, 2:]


def read_seq_file(filepath):
    with open(filepath, "r") as file:
        lines = file.readlines()

    data = []
    for line in lines:
        line_parts = RE_SPACES.split(line)
        if len(line_parts) >= 2 and line_parts[1] == "#":
            data.append(line_parts[2:-1])

    return np.array(data).astype("float64")


def read_ema_file(filepath, format):
    if format == "est":
        return read_est_file(filepath)
    elif format == "seq":
        return read_seq_file(filepath)
    else:
        raise Exception("Uknown EMA format `%s`" % format)


def read_lab_file(filepath, resolution_multiplier=1):
    with open(filepath, "r") as file:
        lines = file.readlines()

    labels = []
    for line in lines:
        match = re.match(RE_LAB_LINE, line)
        assert match is not None
        start, end, name = match.groups()
        start = int(float(start) * resolution_multiplier)
        end = int(float(end) * resolution_multiplier)
        label = {"start": start, "end": end, "name": name}
        labels.append(label)

    return labels


def save_lab_file(filepath, lab):
    file_content = ""
    for label in lab:
        line = "%s %s %s\n" % (label["start"], label["end"], label["name"])
        file_content += line
    with open(filepath, "w") as f:
        f.write(file_content)


def interp_2d(original_data, source_rate, target_rate):
    original_len, ndim = original_data.shape
    resampled_len = round(original_len / source_rate * target_rate)
    xp = np.arange(original_len)
    x = np.linspace(0, original_len - 1, resampled_len, endpoint=True)
    resampled_data = np.zeros((resampled_len, ndim), dtype=original_data.dtype)
    for dim in range(ndim):
        resampled_data[:, dim] = np.interp(x, xp, original_data[:, dim])
    return resampled_data


def parse_item_name(filepath):
    return RE_ITEM_NAME.findall(filepath)[0]


def shuffle_and_split(items, splits_size, seed=None):
    items = items.copy()
    if seed is not None:
        random.seed(seed)
    random.shuffle(items)
    items_len = len(items)

    splits = []
    for split_size in reversed(splits_size[1:]):
        split_len = round(items_len / 100 * split_size)
        split = items[:split_len]
        splits.insert(0, split)
        items = items[split_len:]
    splits.insert(0, items)

    return splits


def create_lowpass_filter(sampling_rate, cutoff_freq, R=0.15):
    fc = cutoff_freq * 2 / sampling_rate
    fc1 = fc + 0.15
    n, wn = signal.cheb1ord(fc, fc1, 0.1, 50)
    C, D = signal.cheby1(n, R, wn)

    def lowpass_filter(x):
        y = x.copy()
        nb_dim = x.shape[1]
        for i_dim in range(nb_dim):
            y[:, i_dim] = signal.filtfilt(C, D, x[:, i_dim])
        return y

    return lowpass_filter


def get_variable_signature(var):
    if type(var) == dict:
        keys = list(var.keys())
        keys.sort()

        signature_items = []
        for key in keys:
            value_signature = get_variable_signature(var[key])
            signature_items.append(f"{key}:{value_signature}")
        signature = "{" + ",".join(signature_items) + "}"
    elif type(var) == list:
        signature_items = []
        for item in var:
            signature_items.append(get_variable_signature(item))
        signature = "[" + ",".join(signature_items) + "]"
    else:
        signature = f"{var}"

    return hashlib.md5(signature.encode()).hexdigest()
