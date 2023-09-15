from torch import nn

from lib.nn.feedforward import FeedForward
from lib.nn.stack_diff import stack_diff


class FeedForwardWithDerivatives(nn.Module):
    def __init__(
        self,
        x_dim,
        nb_derivatives,
        y_dim,
        hidden_dims,
        activation,
        dropout_p,
        batch_norm,
    ):
        super(FeedForwardWithDerivatives, self).__init__()
        self.nb_derivatives = nb_derivatives
        self.feedforward = FeedForward(
            x_dim * (1 + nb_derivatives),
            y_dim,
            hidden_dims,
            activation,
            dropout_p,
            batch_norm,
        )

    def forward(self, x):
        dx = stack_diff(x, self.nb_derivatives)
        return self.feedforward(dx)
