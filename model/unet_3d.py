import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from collections import OrderedDict
from torch import Tensor
from torch.jit.annotations import List
import numpy as np

class unet_3d(nn.Module):
    def __init__(self, config):
        super(unet_3d, self).__init__()
        self.layers = config['layers']
        self.feature_root = config['feature_root']
        self.channels = config['channels']
        self.n_class = config['n_class']
        self.use_bn = config['use_bn']
        self.track_running_stats = config['track_running_stats']
        self.bn_momentum = config['bn_momentum']
        self.conv_repeat = config['conv_repeat']

        if config['loss'] == 'BCE':
            self.loss_func = torch.nn.BCELoss()
        else:
            pass

        self.down_sample_convs = torch.nn.ModuleDict()
        # down sample conv layers
        for layer in range(self.layers):
            feature_number = self.feature_root * (2 ** layer)
            if layer == 0:
                self.down_sample_convs['down{}'.format(layer)] = nn.Sequential(
                    self.get_conv_block(1, feature_number, 'down{}'.format(layer)))
            else:
                self.down_sample_convs['down{}'.format(layer)] = nn.Sequential(
                    self.get_conv_block(feature_number // 2, feature_number, 'down{}'.format(layer)))

        self.up_sample_convs = torch.nn.ModuleDict()
        # up sample conv layers
        for layer in range(self.layers - 2, -1, -1):
            feature_number = self.feature_root * (2 ** layer)
            self.up_sample_convs['up{}'.format(layer)] = nn.Sequential(
                self.get_conv_block(feature_number * 3, feature_number, 'up{}'.format(layer)))


        self.predict_layer = nn.Sequential(OrderedDict([
                ('predict_conv', nn.Conv2d(self.feature_root, self.n_class, kernel_size=3, stride=1, padding=1)),
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
        x = x.unsqueeze(1)
        x = F.max_pool3d(x,kernel_size=2)
        # # convert x from 1x32x1024x1280 to 1x1x32x1024x1280
        down_features = []
        for layer in range(self.layers):
            if layer == 0:
                down_features.append(self.down_sample_convs['down{}'.format(layer)](x))
            else:
                x = F.max_pool3d(down_features[-1], kernel_size=2)
                down_features.append(self.down_sample_convs['down{}'.format(layer)](x))
        up_features = []
        for layer in range(self.layers - 2, -1, -1):
            if layer == self.layers - 2:
                _cat = torch.cat((down_features[layer], F.interpolate(down_features[layer + 1], scale_factor=2)), 1)
            else:
                _cat = torch.cat((down_features[layer], F.interpolate(up_features[-1], scale_factor=2)), 1)
            up_features.append(self.up_sample_convs['up{}'.format(layer)](_cat))

        logits = self.predict_layer(up_features[-1].mean(2))
        logits = F.interpolate(logits,scale_factor=2)
        return logits


    def get_conv_block(self, in_feature, out_feature, prefix):
        _return = OrderedDict()
        for i in range(self.conv_repeat):
            _return[prefix+'_conv{}'.format(i)] = nn.Conv3d(in_feature, out_feature, kernel_size=3, stride=1, padding=1)
            in_feature = out_feature
            if self.use_bn == True:
                _return[prefix+'_norm{}'.format(i)] = nn.BatchNorm3d(out_feature, momentum=self.bn_momentum, track_running_stats=self.track_running_stats)
            _return[prefix + '_relu{}'.format(i)] = nn.ReLU(inplace=True)
        return _return




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