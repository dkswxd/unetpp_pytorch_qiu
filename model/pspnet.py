#!/usr/bin/env python
# coding: utf-8
#
# Author:   Kazuto Nakashima
# URL:      http://kazuto1011.github.io
# Created:  2017-11-15

from __future__ import absolute_import

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .backbone.resnet import _ConvBatchNormReLU, _ResBlock


class _DilatedFCN(nn.Module):
    """ResNet-based Dilated FCN"""

    def __init__(self, n_blocks, channels):
        super(_DilatedFCN, self).__init__()
        self.layer1 = nn.Sequential(
            OrderedDict(
                [
                    ("conv1", _ConvBatchNormReLU(channels, 64, 3, 2, 1, 1)),
                    ("conv2", _ConvBatchNormReLU(64, 64, 3, 1, 1, 1)),
                    ("conv3", _ConvBatchNormReLU(64, 128, 3, 1, 1, 1)),
                    ("pool", nn.MaxPool2d(3, 2, 1)),
                ]
            )
        )
        self.layer2 = _ResBlock(n_blocks[0], 128, 64, 256, 1, 1)
        self.layer3 = _ResBlock(n_blocks[1], 256, 128, 512, 2, 1)
        self.layer4 = _ResBlock(n_blocks[2], 512, 256, 1024, 1, 2)
        self.layer5 = _ResBlock(n_blocks[3], 1024, 512, 2048, 1, 4)

    def forward(self, x):
        h = self.layer1(x)
        h = self.layer2(h)
        h = self.layer3(h)
        h1 = self.layer4(h)
        h2 = self.layer5(h1)
        if self.training:
            return h1, h2
        else:
            return h2


class _PyramidPoolModule(nn.Sequential):
    """Pyramid Pooling Module"""

    def __init__(self, in_channels, pyramids=(6, 3, 2, 1)):
        super(_PyramidPoolModule, self).__init__()
        out_channels = in_channels // len(pyramids)
        self.stages = nn.Module()
        for i, p in enumerate(pyramids):
            self.stages.add_module(
                "s{}".format(i),
                nn.Sequential(
                    OrderedDict(
                        [
                            ("pool", nn.AdaptiveAvgPool2d(output_size=p)),
                            ("conv", _ConvBatchNormReLU(in_channels, out_channels, 1, 1, 0, 1,bn=False)),
                        ]
                    )
                ),
            )

    def forward(self, x):
        hs = [x]
        height, width = x.size()[2:]
        for stage in self.stages.children():
            h = stage(x)
            h = F.upsample(h, (height, width), mode="bilinear")
            hs.append(h)
        return torch.cat(hs, dim=1)


class PSPNet(nn.Module):
    """Pyramid Scene Parsing Network"""

    def __init__(self, config):
        super(PSPNet, self).__init__()
        channels = config['channels']
        self.n_classes = config['n_class']
        n_blocks = config['n_blocks']
        pyramids = config['pyramids']

        if config['loss'] == 'BCE':
            self.loss_func = torch.nn.BCELoss()
        else:
            pass

        self.fcn = _DilatedFCN(n_blocks=n_blocks, channels=channels)
        self.ppm = _PyramidPoolModule(in_channels=2048, pyramids=pyramids)
        # Main branch
        self.final = nn.Sequential(
            OrderedDict(
                [
                    ("pixel_shuffle", nn.PixelShuffle(upscale_factor=8)),
                    ("conv6", nn.Conv2d(4096 // 64, self.n_classes, 1, stride=1, padding=0)),
                    ('softmax', nn.Softmax2d()),
                ]
            )
        )
        # Auxiliary branch
        self.aux = nn.Sequential(
            OrderedDict(
                [
                    ("pixel_shuffle", nn.PixelShuffle(upscale_factor=8)),
                    ("conv6_aux", nn.Conv2d(1024 // 64, self.n_classes, 1, stride=1, padding=0)),
                    ('softmax_aux', nn.Softmax2d()),
                ]
            )
        )

    def forward(self, x):
        if self.training:
            aux, h = self.fcn(x)
            aux = self.aux(aux)
        else:
            h = self.fcn(x)
        h = self.ppm(h)
        h = self.final(h)

        if self.training:
            return [aux, h]
        else:
            return [h]

    def get_loss(self, logits, batch_y):
        loss = torch.tensor(0, dtype=torch.float32).cuda()
        for i in range(len(logits)):
            w = 1 if i == len(logits) - 1 else 0.4
            loss += w * self.loss_func(logits[i], batch_y)
        return loss

    def get_predict(self, logits, thresh=True):
        logits = logits[-1].detach().cpu().numpy()
        pred = logits[0, 1, :, :]
        if thresh:
            pred = np.where(pred > 0.5, 1, 0)
        return pred

    def get_gt(self, batch_y):
        batch_y = batch_y.detach().cpu().numpy()
        batch_y = batch_y[0, 1, :, :]
        return np.where(batch_y > 0.5, 1, 0)