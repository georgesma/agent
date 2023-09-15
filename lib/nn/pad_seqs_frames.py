import torch
import torch.nn.functional as F


def pad_seqs_frames(seqs, pad_size):
    if pad_size == 0:
        return seqs

    seqs_shape = seqs.shape
    seqs_padded = F.pad(seqs[None, ...], (0, 0, pad_size, pad_size), mode="replicate")[
        0
    ]
    # replicate padding not fully implemented for 3d tensors
    strided_shape = (*seqs_shape[:2], 1 + 2 * pad_size, seqs_shape[-1])
    strided_stride = (
        seqs_padded.shape[1] * seqs_shape[2],
        seqs_shape[2],
        seqs_shape[2],
        1,
    )
    seqs_strided = torch.as_strided(seqs_padded, strided_shape, strided_stride)
    seqs_padded = seqs_strided.reshape(*seqs_shape[:-1], -1)

    return seqs_padded


def unpad_seqs_frames(seqs, pad_size):
    if pad_size == 0:
        return seqs

    unpaded_dim = seqs.shape[-1] // (1 + 2 * pad_size)
    start_index = pad_size * unpaded_dim
    end_index = (pad_size + 1) * unpaded_dim

    return seqs[..., start_index:end_index]
