from .ULSTM_seperate import ULSTM_seperate

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from collections import OrderedDict
from torch import Tensor
from torch.jit.annotations import List
import numpy as np

class BiULSTM_seperate(nn.Module):
    def __init__(self, config):
        super(BiULSTM_seperate, self).__init__()

        self.layers = config['layers']
        self.feature_root = config['feature_root']
        self.channels = config['channels']
        self.n_class = config['n_class']
        self.conv_repeat = config['conv_repeat']
        self.use_deform = config['use_deform']

        if config['loss'] == 'BCE':
            self.loss_func = torch.nn.BCELoss()
        else:
            pass

        self.forwardLSTM = ULSTM_seperate(self.layers, self.channels, self.feature_root, self.conv_repeat, self.use_deform)
        self.backwardLSTM = ULSTM_seperate(self.layers, self.channels, self.feature_root, self.conv_repeat, self.use_deform)

        self.predict_layer = nn.Sequential(OrderedDict([
                # ('predict_conv', nn.Conv2d(self.feature_root * 2, self.feature_root * 2, kernel_size=3, stride=1, padding=1)),
                # ('predict_bn', nn.BatchNorm2d(self.feature_root * 2, track_running_stats=False)),
                # ('predict_relu', nn.ReLU(inplace=True)),
                ('predict_conv2', nn.Conv2d(self.feature_root * 2, self.n_class, kernel_size=3, stride=1, padding=1)),
                # ('predict_smax', nn.Sigmoid()),
                ('predict_smax', nn.Softmax2d()),
                ]))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = F.max_pool2d(x, 2, 2)
        B, C, H, W = x.shape
        # x (B, C, H, W)
        x = torch.transpose(x, 0, 1)
        # x (C, B, H, W)
        x = torch.unsqueeze(x, 2)
        # x (C, B, 1, H, W)
        _, forward_h, _ = self.forwardLSTM(x, None, C)
        # forward_h (B, feature_root, H, W)
        _, backward_h, _ = self.backwardLSTM(x.flip((0)), None, C)
        # backward_h (B, feature_root, H, W)
        _cat = torch.cat([forward_h, backward_h], 1)
        logits = self.predict_layer(_cat)
        logits = F.interpolate(logits,scale_factor=2,mode='nearest')
        return logits



    def get_loss(self, logits, batch_y):
        return self.loss_func(logits, batch_y)

    def get_predict(self, logits, thresh=True):
        logits = logits.detach().cpu().numpy()
        pred = logits[0, 1, :, :]
        if thresh:
            pred = np.where(pred > 0.5, 1, 0)
        return pred

    def get_gt(self, batch_y):
        batch_y = batch_y.detach().cpu().numpy()
        batch_y = batch_y[0, 1, :, :]
        return np.where(batch_y > 0.5, 1, 0)