import torch
import pickle
import yaml
import os
from sklearn.preprocessing import StandardScaler

from lib.base_agent import BaseAgent
from lib.sound_speaker_dataloader import get_dataloaders
from lib.nn.simple_lstm import SimpleLSTM
from lib.nn.feedforward_with_derivatives import FeedForwardWithDerivatives
from lib.nn.vq_vae import VQVAE
from lib.nn.loss import compute_jerk_loss

from communicative_agent_nn import CommunicativeAgentNN
from synthesizer.synthesizer import Synthesizer

SYNTHESIZERS_PATH = os.path.join(os.path.dirname(__file__), "../out/synthesizer")


class CommunicativeAgent(BaseAgent):
    def __init__(self, config, load_nn=True):
        self.config = config
        self.nb_speakers = len(self.config["dataset"]["names"])
        self.sound_scaler = StandardScaler()
        self.datasplits = None
        self.synthesizer = Synthesizer.reload(
            "%s/%s" % (SYNTHESIZERS_PATH, config["synthesizer"]["name"]), load_nn=load_nn
        )
        if load_nn:
            self._build_nn(self.config["model"])
            self.nn.eval()

    def _build_nn(self, model_config):
        self.art_dim = self.synthesizer.art_dim
        self.sound_dim = self.synthesizer.sound_dim

        inverse_model = SimpleLSTM(
            model_config["sound_quantizer"]["embedding_dim"],
            self.art_dim,
            model_config["inverse_model"]["hidden_size"],
            model_config["inverse_model"]["num_layers"],
            model_config["inverse_model"]["dropout_p"],
            model_config["inverse_model"]["bidirectional"],
        )

        direct_model = FeedForwardWithDerivatives(
            self.art_dim,
            model_config["direct_model"]["nb_derivatives"],
            self.sound_dim,
            model_config["direct_model"]["hidden_layers"],
            model_config["direct_model"]["activation"],
            model_config["direct_model"]["dropout_p"],
            model_config["direct_model"]["batch_norm"],
        )

        sound_quantizer = VQVAE(
            self.sound_dim,
            model_config["sound_quantizer"]["frame_padding"],
            model_config["sound_quantizer"]["hidden_dims"],
            model_config["sound_quantizer"]["activation"],
            model_config["sound_quantizer"]["embedding_dim"],
            model_config["sound_quantizer"]["num_embeddings"],
            model_config["sound_quantizer"]["commitment_cost"],
            self.nb_speakers,
            model_config["sound_quantizer"]["dropout_p"],
            model_config["sound_quantizer"]["batch_norm"],
        )

        art_quantizer = VQVAE(
            self.art_dim,
            model_config["art_quantizer"]["frame_padding"],
            model_config["art_quantizer"]["hidden_dims"],
            model_config["art_quantizer"]["activation"],
            model_config["art_quantizer"]["embedding_dim"],
            model_config["art_quantizer"]["num_embeddings"],
            model_config["art_quantizer"]["commitment_cost"],
            self.nb_speakers,
            model_config["art_quantizer"]["dropout_p"],
            model_config["art_quantizer"]["batch_norm"],
        )

        self.nn = CommunicativeAgentNN(
            inverse_model, direct_model, sound_quantizer, art_quantizer
        ).to("cuda")

    def get_dataloaders(self):
        datasplits, dataloaders = get_dataloaders(
            self.config["dataset"], self.sound_scaler
        )
        self.datasplits = datasplits
        return dataloaders

    def get_optimizers(self):
        return {
            "inverse_model": torch.optim.Adam(
                self.nn.inverse_model.parameters(),
                lr=self.config["training"]["learning_rate"],
            ),
            "direct_model": torch.optim.Adam(
                self.nn.direct_model.parameters(),
                lr=self.config["training"]["learning_rate"],
            ),
            "sound_quantizer": torch.optim.Adam(
                self.nn.sound_quantizer.parameters(),
                lr=self.config["training"]["learning_rate"],
            ),
            "art_quantizer": torch.optim.Adam(
                self.nn.art_quantizer.parameters(),
                lr=self.config["training"]["learning_rate"],
            ),
        }

    def get_losses_fn(self):
        art_scaler_var = torch.FloatTensor(self.synthesizer.art_scaler.var_).to("cuda")

        def inverse_model_loss(art_seqs_pred, sound_seqs_pred, sound_seqs, seqs_mask):
            reconstruction_error = (sound_seqs_pred - sound_seqs) ** 2
            reconstruction_loss = reconstruction_error[seqs_mask].mean()

            art_seqs_pred = art_seqs_pred * art_scaler_var
            jerk_loss = compute_jerk_loss(art_seqs_pred, seqs_mask)

            total_loss = (
                reconstruction_loss
                + jerk_loss * self.config["training"]["jerk_loss_weight"]
            )

            return total_loss, reconstruction_loss, jerk_loss

        def mse(seqs_pred, seqs, seqs_mask):
            reconstruction_error = (seqs_pred - seqs) ** 2
            reconstruction_loss = reconstruction_error[seqs_mask].mean()
            return reconstruction_loss

        def vq_vae_loss(seqs_pred, seqs, vq_loss_seqs, seqs_mask):
            reconstruction_error = mse(seqs_pred, seqs, seqs_mask)
            vq_loss = vq_loss_seqs[seqs_mask].mean()
            total_loss = reconstruction_error + vq_loss
            return total_loss, reconstruction_error, vq_loss

        return {"inverse_model": inverse_model_loss, "mse": mse, "vq_vae": vq_vae_loss}

    def save(self, save_path):
        with open(save_path + "/config.yaml", "w") as f:
            yaml.safe_dump(self.config, f)
        with open(save_path + "/sound_scaler.pickle", "wb") as f:
            pickle.dump(self.sound_scaler, f)
        with open(save_path + "/datasplits.pickle", "wb") as f:
            pickle.dump(self.datasplits, f)
        torch.save(self.nn.state_dict(), save_path + "/nn_weights.pt")

    @staticmethod
    def reload(save_path, load_nn=True):
        with open(save_path + "/config.yaml", "r") as f:
            config = yaml.safe_load(f)
        agent = CommunicativeAgent(config, load_nn=load_nn)

        with open(save_path + "/sound_scaler.pickle", "rb") as f:
            agent.sound_scaler = pickle.load(f)
        with open(save_path + "/datasplits.pickle", "rb") as f:
            agent.datasplits = pickle.load(f)
        if load_nn:
            agent.nn.load_state_dict(torch.load(save_path + "/nn_weights.pt"))
            agent.nn.eval()

        return agent

    def repeat(self, sound_seq):
        nn_input = torch.FloatTensor(self.sound_scaler.transform(sound_seq)).to("cuda")[
            None, :, :
        ]
        with torch.no_grad():
            _, sound_unit_seq, _, _ = self.nn.sound_quantizer.encode(nn_input)
            art_seq_estimated_unscaled = self.nn.inverse_model(sound_unit_seq)
            sound_seq_estimated_unscaled = self.nn.direct_model(
                art_seq_estimated_unscaled
            )
            _, art_unit_seq, _, _ = self.nn.art_quantizer.encode(
                art_seq_estimated_unscaled
            )

        sound_seq_estimated_unscaled = sound_seq_estimated_unscaled[0].cpu().numpy()
        art_seq_estimated_unscaled = art_seq_estimated_unscaled[0].cpu().numpy()
        art_seq_estimated = self.synthesizer.art_scaler.inverse_transform(
            art_seq_estimated_unscaled
        )
        sound_seq_estimated = self.sound_scaler.inverse_transform(
            sound_seq_estimated_unscaled
        )
        sound_unit_seq = sound_unit_seq[0].cpu().numpy()
        art_unit_seq = art_unit_seq[0].cpu().numpy()

        sound_seq_repeated = self.synthesizer.synthesize(art_seq_estimated)
        return {
            "sound_units": sound_unit_seq,
            "sound_repeated": sound_seq_repeated,
            "sound_estimated": sound_seq_estimated,
            "art_estimated": art_seq_estimated,
            "art_units": art_unit_seq,
        }
