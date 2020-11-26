import pytest
import torch
import torch.nn as nn

from utils.mapping_loss import mapping_loss
from models.mlp import MLP


@pytest.fixture
def batch():
    return 6


@pytest.fixture
def shape():
    return 2, 3, 4


@pytest.fixture
def points1(shape, batch):
    return torch.randn((batch, ) + shape)


@pytest.fixture
def points2(shape, batch):
    return torch.randn((batch, ) + shape)


@pytest.fixture
def poses(batch):
    return torch.randn((batch, ))


def pose_loss(a, b):
    return (a - b) ** 2


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
    mapping_loss(points1, points2, poses, pose_loss)


def test_mlp(use_cuda):

    if use_cuda:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    model = MLP(5, 2, [10, 9, 8], device).to(device)
    print("Model construction completed.")
    model(torch.ones([100, 5]).to(device))
