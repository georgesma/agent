import torch
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

        if load_nn:
            self._build_nn(self.config["model"])
            self.nn.eval()

    def _build_nn(self, model_config):
        self.data_dim = self.main_dataset.get_modality_dim(
            self.config["dataset"]["data_type"]
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
            self.config["dataset"], self.data_scaler
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

    # def synthesize(self, art_seq):
    #     nn_input = torch.FloatTensor(self.art_scaler.transform(art_seq)).to("cuda")
    #     with torch.no_grad():
    #         nn_output = self.nn(nn_input).cpu().numpy()
    #     sound_seq_pred = self.data_scaler.inverse_transform(nn_output)
    #     return sound_seq_pred

    # def synthesize_cuda(self, art_seqs):
    #     with torch.no_grad():
    #         return self.nn(art_seqs)
