import torch
import torch.nn.functional as F
import pickle
import yaml
from sklearn.preprocessing import StandardScaler

from lib import utils
from lib.data_speaker_dataloader import get_dataloaders
from lib.dataset_wrapper import Dataset
from lib.nn.vq_vae import VQVAE


class Quantizer:
    def __init__(self, config, load_nn=True):
        self.config = config
        self.data_scaler = StandardScaler()
        self.datasplits = None
        self.nb_speakers = len(self.config["dataset"]["names"])
        self.main_dataset = Dataset(config["dataset"]["names"][0])

        if "data_type" in self.config["dataset"]:
            self.config["dataset"]["data_types"] = [self.config["dataset"]["data_type"]]
            del self.config["dataset"]["data_type"]

        if load_nn:
            self._build_nn(self.config["model"])
            self.nn.eval()

    def _build_nn(self, model_config):
        self.data_dim = sum(
            [
                self.main_dataset.get_modality_dim(data_type)
                for data_type in self.config["dataset"]["data_types"]
            ]
        )

        self.nn = VQVAE(
            self.data_dim,
            model_config["frame_padding"],
            model_config["hidden_dims"],
            model_config["activation"],
            model_config["embedding_dim"],
            model_config["num_embeddings"],
            model_config["commitment_cost"],
            self.nb_speakers,
            model_config["dropout_p"],
            model_config["batch_norm"],
        ).to("cuda")

    def get_dataloaders(self):
        datasplits, dataloaders = get_dataloaders(
            self.config["dataset"], self.data_scaler, self.datasplits
        )
        self.datasplits = datasplits
        return dataloaders

    def get_optimizer(self):
        return torch.optim.Adam(
            self.nn.parameters(), lr=self.config["training"]["learning_rate"]
        )

    def get_loss_fn(self):
        def mse(seqs_pred, seqs, seqs_mask):
            reconstruction_error = (seqs_pred - seqs) ** 2
            reconstruction_loss = reconstruction_error[seqs_mask].mean()
            return reconstruction_loss

        def loss_fn(seqs_pred, seqs, vq_loss_seqs, seqs_mask):
            reconstruction_error = mse(seqs_pred, seqs, seqs_mask)
            vq_loss = vq_loss_seqs[seqs_mask].mean()
            total_loss = reconstruction_error + vq_loss
            return total_loss, reconstruction_error, vq_loss

        return loss_fn

    def get_signature(self):
        return utils.get_variable_signature(self.config)

    def save(self, save_path):
        with open(save_path + "/config.yaml", "w") as f:
            yaml.safe_dump(self.config, f)
        with open(save_path + "/data_scaler.pickle", "wb") as f:
            pickle.dump(self.data_scaler, f)
        with open(save_path + "/datasplits.pickle", "wb") as f:
            pickle.dump(self.datasplits, f)
        torch.save(self.nn.state_dict(), save_path + "/nn_weights.pt")

    @staticmethod
    def reload(save_path, load_nn=True):
        with open(save_path + "/config.yaml", "r") as f:
            config = yaml.safe_load(f)
        quantizer = Quantizer(config, load_nn=load_nn)

        with open(save_path + "/data_scaler.pickle", "rb") as f:
            quantizer.data_scaler = pickle.load(f)
        with open(save_path + "/datasplits.pickle", "rb") as f:
            quantizer.datasplits = pickle.load(f)

        if load_nn:
            quantizer.nn.load_state_dict(torch.load(save_path + "/nn_weights.pt"))
            quantizer.nn.eval()

        return quantizer

    def autoencode(self, data_seq, speaker_id):
        nn_input = torch.FloatTensor(self.data_scaler.transform(data_seq)).to("cuda")[
            None, :, :
        ]
        data_seq_len = len(data_seq)
        speaker_id = (
            F.one_hot(torch.tensor(speaker_id), num_classes=self.nb_speakers)[None, :]
            .to(torch.float32)
            .repeat(data_seq_len, 1)
            .to("cuda")[None, :, :]
        )
        with torch.no_grad():
            seqs_pred, _, quantized_latent, quantized_index, encoder_output = self.nn(
                nn_input, speaker_id
            )
        autoencoded_features = {
            "seqs_pred": self.data_scaler.inverse_transform(seqs_pred[0].cpu().numpy()),
            "quantized_latent": quantized_latent[0].cpu().numpy(),
            "quantized_index": quantized_index.cpu().numpy(),
            "encoder_output": encoder_output[0].cpu().numpy(),
        }
        return autoencoded_features

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

    def autoencode_datasplit(self, datasplit_index=None):
        quantizer_features = {}
        data_type = self.config["dataset"]["data_type"]

        for dataset_i, dataset_name in enumerate(self.config["dataset"]["names"]):
            dataset_features = {}

            dataset = Dataset(dataset_name)
            if datasplit_index is None:
                items_name = dataset.get_items_name(data_type)
            else:
                items_name = self.datasplits[dataset_name][datasplit_index]

            items_data = dataset.get_items_data(self.config["dataset"]["data_type"])
            for item_name in items_name:
                item_data = items_data[item_name]
                autoencoded = self.autoencode(item_data, dataset_i)
                for autoencoded_type, autoencoded in autoencoded.items():
                    if autoencoded_type not in dataset_features:
                        dataset_features[autoencoded_type] = {}
                    dataset_features[autoencoded_type][item_name] = autoencoded

            quantizer_features[dataset_name] = dataset_features
        return quantizer_features
