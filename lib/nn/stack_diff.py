import torch


def stack_diff(x, nb_diff):
    if nb_diff == 0:
        return x

    x_shape = x.shape
    data_dim = x_shape[-1]
    dx_shape = (*x_shape[:-1], data_dim * (1 + nb_diff))

    dx = torch.zeros(dx_shape, dtype=x.dtype, device=x.device)
    dx[..., :data_dim] = x
    diff = x
    for i in range(nb_diff):
        diff = torch.diff(diff, dim=-2)
        dx[..., (i + 1) :, (i + 1) * data_dim : (i + 2) * data_dim] = diff

    return dx
