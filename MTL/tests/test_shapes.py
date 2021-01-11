import random

import pytest
import torch
import torch.nn as nn

from models.resnet_mtl import BasicBlockMtl

SEED = 42
random.seed(SEED)

def test_basic_block_same_channels():
    random_tensor = torch.rand(1, 3, 224, 224)
    block = BasicBlockMtl(3, 3)
    output = block(random_tensor)
    assert random_tensor.shape == output.shape

def test_basic_block_dif_channels():
    random_tensor = torch.rand(1, 3, 224, 224)
    downsample = nn.Sequential(
        # conv1x1(self.inplanes, planes * block.expansion, stride),
        nn.Conv2d(3, 64 * BasicBlockMtl.expansion, kernel_size=1, stride=1, bias=False),
        nn.BatchNorm2d(64 * BasicBlockMtl.expansion),
    )
    block = BasicBlockMtl(3, 64, downsample=downsample)
    output = block(random_tensor)
    assert output.shape == (1, 64, 224, 224)