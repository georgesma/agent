import torch
import torch.nn.functional as F


def compute_jerk_loss(art_seqs, seqs_mask=None):
    speed_seqs = torch.diff(art_seqs, dim=-2)
    acc_seqs = torch.diff(speed_seqs, dim=-2)
    jerk_seqs = torch.diff(acc_seqs, dim=-2)
    if seqs_mask is not None:
        jerk_seqs = jerk_seqs[seqs_mask[:, 3:]]
    global_jerk = (jerk_seqs ** 2).mean()
    return global_jerk


def ceil_loss(loss, ceil):
    return F.relu(loss - ceil)

