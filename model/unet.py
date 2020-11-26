import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from collections import OrderedDict
from torch import Tensor
from torch.jit.annotations import List

class unet(nn.Module):
    def __init__(self, config):
        super(unet, self).__init__()
        self.layers = config['layers']
        self.feature_root = config['feature_root']
        self.channels = config['channels']
        self.n_class = config['n_class']
        self.use_bn = config['use_bn']
        self.track_running_stats = config['track_running_stats']
        self.bn_momentum = config['bn_momentum']
        self.use_nonlocal = config['use_nonlocal']
        self.conv_repeat = config['conv_repeat']

        self.down_sample_convs = torch.nn.ModuleDict()
        # down sample conv layers
        for layer in range(self.layers):
            feature_number = self.feature_root * (2 ** layer)
            if layer == 0:
                self.down_sample_convs['down{}'.format(layer)] = nn.Sequential(
                    self.get_conv_block(self.channels, feature_number, 'down{}'.format(layer)))
            else:
                od = OrderedDict([('down{}_pool0'.format(layer), nn.MaxPool2d(kernel_size=2))])
                od.update(self.get_conv_block(feature_number // 2, feature_number, 'down{}'.format(layer)))
                self.down_sample_convs['down{}'.format(layer)] = nn.Sequential(od)

        self.up_sample_convs = torch.nn.ModuleDict()
        # up sample conv layers
        for layer in range(self.layers - 2, -1, -1):
            feature_number = self.feature_root * (2 ** layer)
            self.up_sample_convs['up{}'.format(layer)] = nn.Sequential(
                self.get_conv_block(feature_number * 3, feature_number, 'up{}'.format(layer)))

        self.up_sample_transpose = torch.nn.ModuleDict()
        for layer in range(self.layers - 2, -1, -1):
            feature_number = self.feature_root * 2 ** (layer + 1)
            # self.up_sample_transpose['up{}_transpose'.format(layer)] = nn.ConvTranspose2d(feature_number, feature_number, kernel_size=2, stride=2, padding=0)
            self.up_sample_transpose['up{}_transpose'.format(layer)] = nn.UpsamplingNearest2d(scale_factor=2)

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
        down_features = []
        for layer in range(self.layers):
            if layer == 0:
                down_features.append(self.down_sample_convs['down{}'.format(layer)](x))
            else:
                down_features.append(self.down_sample_convs['down{}'.format(layer)](down_features[-1]))
        up_features = []
        for layer in range(self.layers - 2, -1, -1):
            if layer == self.layers - 2:
                _cat = torch.cat((down_features[layer], self.up_sample_transpose['up{}_transpose'.format(layer)](down_features[layer + 1])), 1)
            else:
                _cat = torch.cat((down_features[layer], self.up_sample_transpose['up{}_transpose'.format(layer)](up_features[-1])), 1)
            up_features.append(self.up_sample_convs['up{}'.format(layer)](_cat))
        predict = self.predict_layer(up_features[-1])
        return [predict]


    def get_conv_block(self, in_feature, out_feature, prefix):
        _return = OrderedDict()
        for i in range(self.conv_repeat):
            _return[prefix+'_conv{}'.format(i)] = nn.Conv2d(in_feature, out_feature, kernel_size=3, stride=1, padding=1)
            in_feature = out_feature
            if self.use_bn == True:
                _return[prefix+'_norm{}'.format(i)] = nn.BatchNorm2d(out_feature, momentum=self.bn_momentum, track_running_stats=self.track_running_stats)
            _return[prefix + '_relu{}'.format(i)] = nn.ReLU(inplace=True)
        return _return



