import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from lib import utils
from lib.dataset_wrapper import Dataset


def pad_collate(batch):
    sound_seqs, speaker_seqs, seqs_len, seqs_mask = zip(*batch)

    seqs_len = torch.LongTensor(seqs_len)
    sorted_indices = seqs_len.argsort(descending=True)

    sound_seqs_padded = pad_sequence(sound_seqs, batch_first=True, padding_value=0)
    speaker_seqs_padded = pad_sequence(speaker_seqs, batch_first=True, padding_value=0)
    seqs_mask = pad_sequence(seqs_mask, batch_first=True, padding_value=0)

    return (
        sound_seqs_padded[sorted_indices],
        speaker_seqs_padded[sorted_indices],
        seqs_len[sorted_indices],
        seqs_mask[sorted_indices],
    )


class SoundSpeakerDataset(torch.utils.data.Dataset):
    def __init__(self, sound_seqs, speaker_seqs):
        self.sound_seqs = sound_seqs
        self.speaker_seqs = speaker_seqs
        self.seqs_len = [len(sound_seq) for sound_seq in sound_seqs]
        self.seqs_mask = [torch.BoolTensor([1] * seq_len) for seq_len in self.seqs_len]
        self.len = len(sound_seqs)

    def __getitem__(self, idx):
        sound_seq = self.sound_seqs[idx]
        speaker_seq = self.speaker_seqs[idx]
        seq_len = self.seqs_len[idx]
        seq_mask = self.seqs_mask[idx]
        return sound_seq, speaker_seq, seq_len, seq_mask

    def __len__(self):
        return self.len


def get_dataloaders(dataset_config, sound_scaler, datasplits):
    datasets_sound_data = {}
    if datasplits is None:
        datasplits = {}
    nb_datasets = len(dataset_config["names"])

    for dataset_name in dataset_config["names"]:
        dataset = Dataset(dataset_name)
        dataset_sound_data = dataset.get_items_data(
            dataset_config["sound_type"], cut_silences=True
        )
        dataset_items_name = list(dataset_sound_data.keys())
        if dataset_name not in datasplits:
            dataset_datasplits = utils.shuffle_and_split(
                dataset_items_name, dataset_config["datasplits_size"]
            )
            datasplits[dataset_name] = dataset_datasplits

        datasets_sound_data[dataset_name] = dataset_sound_data

    dataloaders = []

    for i_datasplit in range(3):
        split_sound_seqs = []
        split_speaker_seqs = []
        for i_dataset, (dataset_name, dataset_sound_data) in enumerate(
            datasets_sound_data.items()
        ):
            dataset_split_items = datasplits[dataset_name][i_datasplit]
            dataset_sound_data = datasets_sound_data[dataset_name]
            dataset_split_sound_seqs = [
                dataset_sound_data[split_item] for split_item in dataset_split_items
            ]
            split_sound_seqs += dataset_split_sound_seqs

            speaker = F.one_hot(torch.tensor(i_dataset), num_classes=nb_datasets)[
                None, :
            ].to(torch.float32)
            split_speaker_seqs += [
                speaker.repeat(len(sound_seq), 1) for sound_seq in dataset_split_sound_seqs
            ]

        if i_datasplit == 0:
            split_sound_concat = np.concatenate(split_sound_seqs)
            sound_scaler.fit(split_sound_concat)

        split_sound_seqs = [
            torch.FloatTensor(sound_scaler.transform(split_sound_seq))
            for split_sound_seq in split_sound_seqs
        ]

        split_dataloader = torch.utils.data.DataLoader(
            SoundSpeakerDataset(split_sound_seqs, split_speaker_seqs),
            batch_size=dataset_config["batch_size"],
            shuffle=dataset_config["shuffle_between_epochs"],
            num_workers=dataset_config["num_workers"],
            collate_fn=pad_collate,
        )
        dataloaders.append(split_dataloader)

    return datasplits, dataloaders
