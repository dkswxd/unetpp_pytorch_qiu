import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from collections import OrderedDict
from torch import Tensor
from torch.jit.annotations import List
import numpy as np
from .CLSTM import ConvLSTM

class BiCLSTM_in_unet(nn.Module):
    def __init__(self, config):
        super(BiCLSTM_in_unet, self).__init__()
        self.layers = config['layers']
        self.channels = config['channels']
        self.n_class = config['n_class']

        self.BiCLSTM_feature_root = config['BiCLSTM_feature_root']
        if config['loss'] == 'BCE':
            self.loss_func = torch.nn.BCELoss()
        else:
            pass


        self.down_sample_convs = torch.nn.ModuleDict()
        # down sample conv layers
        for layer in range(self.layers):
            feature_number = self.BiCLSTM_feature_root * (2 ** layer)
            if layer == 0:
                self.down_sample_convs['down{}'.format(layer)] = BiCLSTM_block(1, feature_number)
            else:
                self.down_sample_convs['down{}'.format(layer)] = BiCLSTM_block(feature_number, feature_number)

        self.up_sample_convs = torch.nn.ModuleDict()
        # up sample conv layers
        for layer in range(self.layers - 2, -1, -1):
            feature_number = self.BiCLSTM_feature_root * (2 ** layer)
            self.up_sample_convs['up{}'.format(layer)] = BiCLSTM_block(feature_number * 6, feature_number)


        self.predict_layer = nn.Sequential(OrderedDict([
                ('predict_conv', nn.Conv2d(self.BiCLSTM_feature_root * 2, self.n_class, kernel_size=3, stride=1, padding=1)),
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
        # ->
        # x (S, B, C, H, W)

        down_features = []
        for layer in range(self.layers):
            if layer == 0:
                down_features.append(self.down_sample_convs['down{}'.format(layer)](x))
            else:
                down_features.append(self.down_sample_convs['down{}'.format(layer)](F.max_pool3d(down_features[-1], (1, 2, 2), (1, 2, 2))))

        up_features = []
        for layer in range(self.layers - 2, -1, -1):
            if layer == self.layers - 2:
                _cat = torch.cat((down_features[layer], F.interpolate(down_features[layer + 1], scale_factor=(1, 2, 2), mode='nearest')), 2)
            else:
                _cat = torch.cat((down_features[layer], F.interpolate(up_features[-1], scale_factor=(1, 2, 2), mode='nearest')), 2)
            up_features.append(self.up_sample_convs['up{}'.format(layer)](_cat))

        logits = self.predict_layer(up_features[-1][-1, ...])
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


####################
class BiCLSTM_block(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(BiCLSTM_block, self).__init__()

        self.forwardLSTM = ConvLSTM(in_channel, out_channel, 3, stride=1, padding=1)
        self.backwardLSTM = ConvLSTM(in_channel, out_channel, 3, stride=1, padding=1)

    def forward(self, x):
        # x (S, B, C, H, W)
        S = x.shape[0]
        forward_outputs, forward_h, forward_c = self.forwardLSTM(x, None, S)
        backward_outputs, backward_h, backward_c = self.backwardLSTM(x.flip((0)), None, S)
        _cat = torch.cat([forward_outputs, backward_outputs], 2)
        return _cat