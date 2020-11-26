import pytest
import torch
import torch.nn as nn

from utils.mapping_loss import mapping_loss
from models.mlp import MLP


@pytest.fixture
def num():
    return 23


@pytest.fixture
def shape():
    return 2, 3, 4


@pytest.fixture
def points1(num, shape):
    return [torch.randn(shape) for _ in range(num)]


@pytest.fixture
def points2(num, shape):
    return [torch.randn(shape) for _ in range(num)]


@pytest.fixture
def poses(num):
    return [torch.randn(()) for _ in range(num)]


def pose_loss(a, b):
    return (a - b) ** 2


def test_mapping_loss(num, points1, points2, poses):
    mapping_loss(points1, points2, poses, pose_loss)


def test_mlp():
    model = MLP(5, 2, [4, 4])
    model(torch.ones([100, 5]))
