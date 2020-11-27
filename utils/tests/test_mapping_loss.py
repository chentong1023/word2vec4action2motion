import pytest
import torch
import torch.nn as nn

from utils.mapping_loss import mapping_loss
from models.mlp import MLP


@pytest.fixture
def batch():
    return 10


@pytest.fixture
def point_shape():
    return 2, 3, 4


@pytest.fixture
def pose_shape():
    return 5, 6, 7, 8


@pytest.fixture
def points1(point_shape, batch):
    return torch.randn((batch,) + point_shape)


@pytest.fixture
def points2(point_shape, batch):
    return torch.randn((batch,) + point_shape)


@pytest.fixture
def poses(batch, pose_shape):
    return torch.randn((batch,) + pose_shape)


def pose_loss(a, b):
    return torch.sum((a - b) ** 2)


@pytest.fixture
def use_cuda():
    return True


def test_mapping_loss(points1, points2, poses, use_cuda):
    assert isinstance(points1, torch.Tensor)
    assert isinstance(points2, torch.Tensor)
    assert isinstance(poses, torch.Tensor)
    if use_cuda:
        points1 = points1.cuda()
        points2 = points2.cuda()
        poses = poses.cuda()

    if use_cuda:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    loss1, loss2 = mapping_loss(points1, points2, poses, pose_loss, device)
    assert loss1.shape == () == loss2.shape


def test_mlp(use_cuda):

    if use_cuda:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    model = MLP(5, 2, [10, 9, 8]).to(device)
    print("Model construction completed.")
    model(torch.ones([100, 5]).to(device))
