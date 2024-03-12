from torch import nn
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"


class FeedForward(nn.Module):
    def __init__(self, x_dim, y_dim, hidden_dims, activation, dropout_p, batch_norm):
        super(FeedForward, self).__init__()
        self.build(x_dim, y_dim, hidden_dims, activation, dropout_p, batch_norm)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def build(self, x_dim, y_dim, hidden_dims, activation, dropout_p, batch_norm):
        layers = []
        prev_dim = x_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))

            if batch_norm is True:
                layers.append(nn.BatchNorm1d(hidden_dim))

            if activation == "leaky_relu":
                layers.append(nn.LeakyReLU())
            elif activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "tanh":
                layers.append(nn.Tanh())
            else:
                raise Exception("Unknown activation function '%s'" % activation)

            if dropout_p > 0:
                layers.append(nn.Dropout(p=dropout_p))

            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, y_dim))
        self.nn = nn.Sequential(*layers)

    def forward(self, x):
        x = x.to(self.device)  # Ensure input is on the correct device

        original_shape = x.shape
        if len(original_shape) == 3:
            x = x.reshape(-1, original_shape[-1])
        y = self.nn(x)
        if len(original_shape) == 3:
            y = y.reshape(*original_shape[:-1], -1)
        return y
