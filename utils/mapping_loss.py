import torch
import torch.nn as n
import torch.nn.functional as F
from typing import List

__all__ = ["mapping_loss"]


def loss_matrix(poses: torch.Tensor, pose_loss):
    pose_n = poses.shape[0]
    loss_mat = torch.empty([pose_n, pose_n])
    for i in range(pose_n):
        for j in range(pose_n):
            loss_mat[i, j] = pose_loss(poses[i], poses[j])
    return loss_mat


def mapping_loss(original_p: torch.Tensor, mapped_p: torch.Tensor, poses: torch.Tensor, pose_loss):

    assert original_p.shape[0] == mapped_p.shape[0] == poses.shape[0]
    num = original_p.shape[0]

    # origin_p_tensor = torch.stack(original_p)
    # mapped_p_tensor = torch.stack(mapped_p)
    similarity_loss = (original_p - mapped_p) ** 2

    triplet_loss = torch.tensor(0.)
    loss_mat = loss_matrix(poses, pose_loss)
    for i in range(num):
        for j in range(num):
            for k in range(j + 1, num):
                if i == j or i == k:
                    continue
                loss_i_j = loss_mat[i, j]
                loss_i_k = loss_mat[i, k]
                dis_i_j = (mapped_p[i] - mapped_p[j]) ** 2
                dis_i_k = (mapped_p[i] - mapped_p[k]) ** 2

                if loss_i_j < loss_i_k:
                    triplet_loss += torch.sum(torch.max(dis_i_j - dis_i_k, torch.tensor(0.)))
                else:
                    triplet_loss += torch.sum(torch.max(dis_i_k - dis_i_j, torch.tensor(0.)))

    return similarity_loss, triplet_loss
