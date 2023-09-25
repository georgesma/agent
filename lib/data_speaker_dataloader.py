import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from lib import utils
from lib.dataset_wrapper import Dataset


def pad_collate(batch):
    data_seqs, speaker_seqs, seqs_len, seqs_mask = zip(*batch)

    seqs_len = torch.LongTensor(seqs_len)
    sorted_indices = seqs_len.argsort(descending=True)

    data_seqs_padded = pad_sequence(data_seqs, batch_first=True, padding_value=0)
    speaker_seqs_padded = pad_sequence(speaker_seqs, batch_first=True, padding_value=0)
    seqs_mask = pad_sequence(seqs_mask, batch_first=True, padding_value=0)

    return (
        data_seqs_padded[sorted_indices],
        speaker_seqs_padded[sorted_indices],
        seqs_len[sorted_indices],
        seqs_mask[sorted_indices],
    )


class DataSpeakerDataset(torch.utils.data.Dataset):
    def __init__(self, data_seqs, speaker_seqs):
        self.data_seqs = data_seqs
        self.speaker_seqs = speaker_seqs
        self.seqs_len = [len(data_seq) for data_seq in data_seqs]
        self.seqs_mask = [torch.BoolTensor([1] * seq_len) for seq_len in self.seqs_len]
        self.len = len(data_seqs)

    def __getitem__(self, idx):
        data_seq = self.data_seqs[idx]
        speaker_seq = self.speaker_seqs[idx]
        seq_len = self.seqs_len[idx]
        seq_mask = self.seqs_mask[idx]
        return data_seq, speaker_seq, seq_len, seq_mask

    def __len__(self):
        return self.len


def get_dataloaders(dataset_config, data_scaler):
    datasets_data = {}
    datasplits = {}
    nb_datasets = len(dataset_config["names"])

    for dataset_name in dataset_config["names"]:
        dataset = Dataset(dataset_name)
        dataset_items_data = dataset.get_items_data(
            dataset_config["data_type"], cut_silences=True
        )
        dataset_items_name = list(dataset_items_data.keys())
        dataset_datasplits = utils.shuffle_and_split(
            dataset_items_name, dataset_config["datasplits_size"]
        )

        datasets_data[dataset_name] = dataset_items_data
        datasplits[dataset_name] = dataset_datasplits

    dataloaders = []

    for i_datasplit in range(3):
        split_data_seqs = []
        split_speaker_seqs = []
        for i_dataset, (dataset_name, dataset_items_data) in enumerate(
            datasets_data.items()
        ):
            dataset_split_items = datasplits[dataset_name][i_datasplit]
            dataset_items_data = datasets_data[dataset_name]
            dataset_split_data_seqs = [
                dataset_items_data[split_item] for split_item in dataset_split_items
            ]
            split_data_seqs += dataset_split_data_seqs

            speaker = F.one_hot(torch.tensor(i_dataset), num_classes=nb_datasets)[
                None, :
            ].to(torch.float32)
            split_speaker_seqs += [
                speaker.repeat(len(data_seq), 1) for data_seq in dataset_split_data_seqs
            ]

        if i_datasplit == 0:
            split_data_concat = np.concatenate(split_data_seqs)
            data_scaler.fit(split_data_concat)

        split_data_seqs = [
            torch.FloatTensor(data_scaler.transform(split_data_seq))
            for split_data_seq in split_data_seqs
        ]

        split_dataloader = torch.utils.data.DataLoader(
            DataSpeakerDataset(split_data_seqs, split_speaker_seqs),
            batch_size=dataset_config["batch_size"],
            shuffle=dataset_config["shuffle_between_epochs"],
            num_workers=dataset_config["num_workers"],
            collate_fn=pad_collate,
        )
        dataloaders.append(split_dataloader)

    return datasplits, dataloaders
