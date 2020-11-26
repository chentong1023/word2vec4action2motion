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


def test_mapping_loss(points1, points2, poses):
    assert isinstance(points1, torch.Tensor)
    assert isinstance(points2, torch.Tensor)
    assert isinstance(poses, torch.Tensor)
    mapping_loss(points1, points2, poses, pose_loss)


def test_mlp():
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    model = MLP(5, 2, [10, 9, 8], device).to(device)
    print("Model construction completed.")
    model(torch.ones([100, 5]).to(device))
