import torch
import pickle
import yaml
from sklearn.preprocessing import StandardScaler

from lib import utils
from lib.art_sound_dataloader import get_dataloaders
from lib.dataset_wrapper import Dataset
from lib.nn.feedforward import FeedForward


class Synthesizer:
    def __init__(self, config, load_nn=True):
        self.config = config
        self.sound_scaler = StandardScaler()
        self.art_scaler = StandardScaler()
        self.datasplits = None
        self.dataset = Dataset(config["dataset"]["name"])

        if load_nn:
            self._build_nn(self.config["model"])
            self.nn.eval()

    def _build_nn(self, model_config):
        self.art_dim = (
            self.dataset.get_modality_dim(self.config["dataset"]["art_type"])
        )
        self.sound_dim = self.dataset.get_modality_dim(
            self.config["dataset"]["sound_type"]
        )

        self.nn = FeedForward(
            self.art_dim,
            self.sound_dim,
            model_config["hidden_layers"],
            model_config["activation"],
            model_config["dropout_p"],
            model_config["batch_norm"],
        ).to("cuda")

    def get_dataloaders(self):
        datasplits, dataloaders = get_dataloaders(
            self.config["dataset"], self.art_scaler, self.sound_scaler
        )
        self.datasplits = datasplits
        return dataloaders

    def get_optimizer(self):
        return torch.optim.Adam(
            self.nn.parameters(), lr=self.config["training"]["learning_rate"]
        )

    def get_loss_fn(self):
        def loss_fn(sound_seqs_pred, sound_seqs, seqs_mask):
            reconstruction_error = (sound_seqs_pred - sound_seqs) ** 2
            reconstruction_loss = reconstruction_error[seqs_mask].mean()
            return reconstruction_loss

        return loss_fn

    def get_signature(self):
        return utils.get_variable_signature(self.config)

    def save(self, save_path):
        with open(save_path + "/config.yaml", "w") as f:
            yaml.safe_dump(self.config, f)
        with open(save_path + "/sound_scaler.pickle", "wb") as f:
            pickle.dump(self.sound_scaler, f)
        with open(save_path + "/art_scaler.pickle", "wb") as f:
            pickle.dump(self.art_scaler, f)
        with open(save_path + "/datasplits.pickle", "wb") as f:
            pickle.dump(self.datasplits, f)
        torch.save(self.nn.state_dict(), save_path + "/nn_weights.pt")

    @staticmethod
    def reload(save_path, load_nn=True):
        with open(save_path + "/config.yaml", "r") as f:
            config = yaml.safe_load(f)
        synthesizer = Synthesizer(config, load_nn=load_nn)

        with open(save_path + "/sound_scaler.pickle", "rb") as f:
            synthesizer.sound_scaler = pickle.load(f)
        with open(save_path + "/art_scaler.pickle", "rb") as f:
            synthesizer.art_scaler = pickle.load(f)
        with open(save_path + "/datasplits.pickle", "rb") as f:
            synthesizer.datasplits = pickle.load(f)

        if load_nn:
            synthesizer.nn.load_state_dict(torch.load(save_path + "/nn_weights.pt"))
            synthesizer.nn.eval()

        return synthesizer

    def synthesize(self, art_seq):
        nn_input = torch.FloatTensor(self.art_scaler.transform(art_seq)).to("cuda")
        with torch.no_grad():
            nn_output = self.nn(nn_input).cpu().numpy()
        sound_seq_pred = self.sound_scaler.inverse_transform(nn_output)
        return sound_seq_pred

    def synthesize_cuda(self, art_seqs):
        with torch.no_grad():
            return self.nn(art_seqs)
