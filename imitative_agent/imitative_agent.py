import torch
import pickle
import yaml
import os,sys
from sklearn.preprocessing import StandardScaler
print("current path:", os.getcwd())
# sys.path.insert(0, "/Users/ladislas/Desktop/motor_control_agent")
sys.path.insert(0, "/mnt/c/Users/vpaul/Documents/Inner_Speech/agent/")

from lib.base_agent import BaseAgent
from lib.sound_dataloader import get_dataloaders
from lib.nn.simple_lstm import SimpleLSTM
from lib.nn.feedforward import FeedForward
from lib.nn.loss import ceil_loss, compute_jerk_loss

from imitative_agent_nn import ImitativeAgentNN
from synthesizer.synthesizer import Synthesizer

SYNTHESIZERS_PATH = os.path.join(os.path.dirname(__file__), "../out/synthesizer")


class ImitativeAgent(BaseAgent):
    def __init__(self, config, load_nn=True):
        self.config = config
        self.sound_scaler = StandardScaler()
        self.datasplits = None
        #self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if load_nn:
            print(self.config["model"])

            self.synthesizer = Synthesizer.reload(
                "%s/%s" % (SYNTHESIZERS_PATH, config["synthesizer"]["name"])
            )
            print(self.config["model"])
            self._build_nn(self.config["model"])
            self.nn.eval()

    def _build_nn(self, model_config):
        self.art_dim = self.synthesizer.art_dim
        self.sound_dim = self.synthesizer.sound_dim

        inverse_model = SimpleLSTM(
            self.sound_dim,
            self.art_dim,
            model_config["inverse_model"]["hidden_size"],
            model_config["inverse_model"]["num_layers"],
            model_config["inverse_model"]["dropout_p"],
            model_config["inverse_model"]["bidirectional"],
        )

        direct_model = FeedForward(
            self.art_dim,
            self.sound_dim,
            model_config["direct_model"]["hidden_layers"],
            model_config["direct_model"]["activation"],
            model_config["direct_model"]["dropout_p"],
            model_config["direct_model"]["batch_norm"],
        )

        self.nn = ImitativeAgentNN(inverse_model, direct_model)  #.to(self.device)

    def get_dataloaders(self):
        datasplits, dataloaders = get_dataloaders(
            self.config["dataset"], self.sound_scaler, self.datasplits
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
        }

    def get_losses_fn(self):
        art_scaler_var = torch.FloatTensor(self.synthesizer.art_scaler.var_)  #.to(self.device)

        def inverse_model_loss(art_seqs_pred, sound_seqs_pred, sound_seqs, seqs_mask):
            reconstruction_error = (sound_seqs_pred - sound_seqs) ** 2
            reconstruction_loss = reconstruction_error[seqs_mask].mean()

            art_seqs_pred = art_seqs_pred * art_scaler_var
            jerk_loss = compute_jerk_loss(art_seqs_pred, seqs_mask)

            total_loss = reconstruction_loss + (
                ceil_loss(jerk_loss, self.config["training"]["jerk_loss_ceil"])
                * self.config["training"]["jerk_loss_weight"]
            )

            return total_loss, reconstruction_loss, jerk_loss

        def mse(sound_seqs_pred, sound_seqs, seqs_mask):
            reconstruction_error = (sound_seqs_pred - sound_seqs) ** 2
            reconstruction_loss = reconstruction_error[seqs_mask].mean()
            return reconstruction_loss

        return {"inverse_model": inverse_model_loss, "mse": mse}

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
            print(config)
        agent = ImitativeAgent(config)

        with open(save_path + "/sound_scaler.pickle", "rb") as f:
            agent.sound_scaler = pickle.load(f)
        with open(save_path + "/datasplits.pickle", "rb") as f:
            agent.datasplits = pickle.load(f)
        if load_nn:
            agent.nn.load_state_dict(torch.load(save_path + "/nn_weights.pt"))
            agent.nn.eval()

        return agent

    def repeat(self, sound_seq):
        nn_input = torch.FloatTensor(self.sound_scaler.transform(sound_seq))  #.to(self.device)
        with torch.no_grad():
            sound_seq_estimated_unscaled, art_seq_estimated_unscaled = self.nn(
                nn_input[None, :, :]
            )
        sound_seq_estimated_unscaled = sound_seq_estimated_unscaled[0].cpu().numpy()
        art_seq_estimated_unscaled = art_seq_estimated_unscaled[0].cpu().numpy()
        art_seq_estimated = self.synthesizer.art_scaler.inverse_transform(
            art_seq_estimated_unscaled
        )
        sound_seq_estimated = self.sound_scaler.inverse_transform(
            sound_seq_estimated_unscaled
        )
        sound_seq_repeated = self.synthesizer.synthesize(art_seq_estimated)
        return {
            "sound_repeated": sound_seq_repeated,
            "sound_estimated": sound_seq_estimated,
            "art_estimated": art_seq_estimated,
        }

    # def invert_art(self, sound_seq):
    #     nn_input = torch.FloatTensor(self.sound_scaler.transform(sound_seq)).to("cuda")
    #     with torch.no_grad():
    #         art_seq_estimated_unscaled = self.nn.inverse_model(
    #             nn_input[None, :, :]
    #         )
    #     art_seq_estimated_unscaled = art_seq_estimated_unscaled[0].cpu().numpy()
    #     art_seq_estimated = self.synthesizer.art_scaler.inverse_transform(
    #         art_seq_estimated_unscaled
    #     )
    #     return art_seq_estimated
