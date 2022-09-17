# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license https://github.com/facebookresearch/vissl/blob/main/LICENSE
# simplified version of  https://github.com/facebookresearch/vissl/blob/main/vissl/models/trunks/resnext.py

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.resnet import BasicBlock, Bottleneck

# For more depths, add the block config here
BLOCK_CONFIG = {
    18: {"layers": (2, 2, 2, 2), "block": BasicBlock},
    34: {"layers": (3, 4, 6, 3), "block": BasicBlock},
    50: {"layers": (3, 4, 6, 3), "block": Bottleneck},
    101: {"layers": (3, 4, 23, 3), "block": Bottleneck},
    152: {"layers": (3, 8, 36, 3), "block": Bottleneck},
    200: {"layers": (3, 24, 36, 3), "block": Bottleneck},
}


class ResNeXt(nn.Module):
    """
    Wrapper for TorchVison ResNet Model to support different depth and
    width_multiplier. We provide flexibility with LAB input, stride in last
    ResNet block and type of norm (BatchNorm, LayerNorm)
    """

    def __init__(self, width_per_group=64, depth=50, width_multiplier=1, input_channels=3):
        super(ResNeXt, self).__init__()

        self.depth = depth
        self.width_multiplier = width_multiplier
        self._norm_layer = nn.BatchNorm2d
        self.width_per_group = width_per_group
        self.input_channels = input_channels

        (n1, n2, n3, n4) = BLOCK_CONFIG[self.depth]["layers"]
        block_constructor = BLOCK_CONFIG[self.depth]["block"]

        model = models.resnet.ResNet(
            block=block_constructor,
            layers=[n1, n2, n3, n4],
            zero_init_residual=False,
            groups=1,
            width_per_group=self.width_per_group,
            norm_layer=self._norm_layer,
        )

        model.inplanes = 64 * self.width_multiplier
        dim_inner = 64 * self.width_multiplier

        model_conv1 = nn.Conv2d(
            self.input_channels,
            model.inplanes,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        model_bn1 = self._norm_layer(model.inplanes)
        model_relu1 = model.relu
        model_maxpool = model.maxpool
        model_avgpool = model.avgpool
        model_layer1 = model._make_layer(block_constructor, dim_inner, n1)
        model_layer2 = model._make_layer(block_constructor, dim_inner * 2, n2, stride=2)
        model_layer3 = model._make_layer(block_constructor, dim_inner * 4, n3, stride=2)
        model_layer4 = model._make_layer(
            block_constructor, dim_inner * 8, n4, stride=2
        )
        self._feature_blocks = nn.ModuleDict(
            [
                ("conv1", model_conv1),
                ("bn1", model_bn1),
                ("conv1_relu", model_relu1),
                ("maxpool", model_maxpool),
                ("layer1", model_layer1),
                ("layer2", model_layer2),
                ("layer3", model_layer3),
                ("layer4", model_layer4),
                ("avgpool", model_avgpool),
                ("flatten", nn.Flatten(1)),
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        for feature_name, feature_block in self._feature_blocks.items():
            # The last chunk has to be non-volatile
            x = feature_block(x)

        return x