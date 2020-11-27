import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

__all__ = ["mapping_loss"]


def loss_matrix(poses: torch.Tensor, pose_loss, device) -> torch.Tensor:
    batch_size = poses.shape[0]
    loss_mat = torch.empty([batch_size, batch_size]).to(device)
    for i in range(batch_size):
        for j in range(batch_size):
            loss_mat[i, j] = pose_loss(poses[i], poses[j])
    assert loss_mat[0, 0].shape == ()
    return loss_mat

def dis_matrix(original_p: torch.Tensor, mapped_p: torch.Tensor, device) -> torch.Tensor:
    batch_size = original_p.shape[0]
    dis_mat = torch.empty([batch_size, batch_size]).to(device)
    for i in range(batch_size):
        for j in range(batch_size):
            dis_mat[i, j] = torch.sum((mapped_p[i] - mapped_p[j]) ** 2)
    assert dis_mat[0, 0].shape == ()
    return dis_mat

def mapping_loss(
    original_p: torch.Tensor, mapped_p: torch.Tensor, poses: torch.Tensor, pose_loss, device
) -> Tuple[torch.Tensor, torch.Tensor]:

    assert original_p.shape[0] == mapped_p.shape[0] == poses.shape[0]
    batch_size = original_p.shape[0]

    similarity_loss = torch.sum((original_p - mapped_p) ** 2)

    triplet_loss = torch.tensor(0.0).to(device)
    loss_mat = loss_matrix(poses, pose_loss, device)
    dis_mat = dis_matrix(original_p, mapped_p, device)

    relu = nn.ReLU()

    for i in range(batch_size):
        for j in range(batch_size):
            for k in range(batch_size):
                if loss_mat[i, j] < loss_mat[i, k]:
                    triplet_loss += relu(dis_mat[i, j] - dis_mat[i, k])
                else:
                    triplet_loss += relu(dis_mat[i, k] - dis_mat[i, j])

    assert similarity_loss.shape == () == triplet_loss.shape
    return similarity_loss, triplet_loss
