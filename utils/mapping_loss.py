import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

__all__ = ["mapping_loss"]


def loss_matrix(poses: torch.Tensor, pose_loss) -> torch.Tensor:
    batch_size = poses.shape[0]
    loss_mat = torch.empty([batch_size, batch_size])
    for i in range(batch_size):
        for j in range(batch_size):
            loss_mat[i, j] = pose_loss(poses[i], poses[j])
    assert loss_mat[0, 0].shape == ()
    return loss_mat


def mapping_loss(
    original_p: torch.Tensor, mapped_p: torch.Tensor, poses: torch.Tensor, pose_loss
) -> Tuple[torch.Tensor, torch.Tensor]:

    assert original_p.shape[0] == mapped_p.shape[0] == poses.shape[0]
    batch_size = original_p.shape[0]

    similarity_loss = torch.sum((original_p - mapped_p) ** 2)

    triplet_loss = torch.tensor(0.0)
    loss_mat = loss_matrix(poses, pose_loss)
    relu = nn.ReLU()
    for i in range(batch_size):
        for j in range(batch_size):
            for k in range(j + 1, batch_size):
                loss_i_j = loss_mat[i, j]
                loss_i_k = loss_mat[i, k]
                dis_i_j = torch.sum((mapped_p[i] - mapped_p[j]) ** 2)
                dis_i_k = torch.sum((mapped_p[i] - mapped_p[k]) ** 2)

                if loss_i_j < loss_i_k:
                    triplet_loss += relu(dis_i_j - dis_i_k)
                else:
                    triplet_loss += relu(dis_i_k - dis_i_j)

    return similarity_loss, triplet_loss
