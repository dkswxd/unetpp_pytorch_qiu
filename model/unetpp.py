import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from collections import OrderedDict
from torch import Tensor
from torch.jit.annotations import List
import numpy as np

class unetpp(nn.Module):
    def __init__(self, config):
        super(unetpp, self).__init__()
        self.layers = config['layers']
        self.feature_root = config['feature_root']
        self.channels = config['channels']
        self.n_class = config['n_class']
        self.use_bn = config['use_bn']
        self.track_running_stats = config['track_running_stats']
        self.bn_momentum = config['bn_momentum']
        self.use_gn = config['use_gn']
        self.num_groups = config['num_groups']
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
                    self.get_conv_block(self.channels, feature_number, 'down{}'.format(layer)))
            else:
                od = OrderedDict([('down{}_pool0'.format(layer), nn.MaxPool2d(kernel_size=2))])
                od.update(self.get_conv_block(feature_number // 2, feature_number, 'down{}'.format(layer)))
                self.down_sample_convs['down{}'.format(layer)] = nn.Sequential(od)


        self.up_sample_convs = torch.nn.ModuleDict()
        # up sample conv layers
        for layer_i in range(1, self.layers):
            for layer_j in range(self.layers - layer_i):
                feature_number = self.feature_root * (2 ** layer_j)
                self.up_sample_convs['up{}_{}'.format(layer_i, layer_j)] = nn.Sequential(
                    self.get_conv_block(feature_number * (2 + layer_i), feature_number, 'up{}_{}'.format(layer_i, layer_j)))


        # up sample layers
        self.up_sample_transpose = torch.nn.ModuleDict()
        for layer_i in range(1, self.layers):
            for layer_j in range(self.layers - layer_i):
                feature_number = self.feature_root * 2 ** (layer_j + 1)
                # self.up_sample_transpose['up{}_transpose'.format(layer)] = nn.ConvTranspose2d(feature_number, feature_number, kernel_size=2, stride=2, padding=0)
                self.up_sample_transpose['up{}_{}_transpose'.format(layer_i, layer_j)] = nn.UpsamplingNearest2d(scale_factor=2)

        # up predict layer
        self.predict_layer = torch.nn.ModuleDict()
        for layer_i in range(1, self.layers):
            self.predict_layer['predict{}'.format(layer_i)] = nn.Sequential(OrderedDict([
                ('predict{}_conv'.format(layer_i), nn.Conv2d(self.feature_root, self.n_class, kernel_size=3, stride=1, padding=1)),
                # ('predict_smax', nn.Sigmoid()),
                ('predict{}_smax'.format(layer_i), nn.Softmax2d()),
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
                down_features.append([self.down_sample_convs['down{}'.format(layer)](x)])
            else:
                down_features.append([self.down_sample_convs['down{}'.format(layer)](down_features[-1][0])])
        # up_features = []
        for layer_i in range(1, self.layers):
            for layer_j in range(self.layers - layer_i):
                _cat = [self.up_sample_transpose['up{}_{}_transpose'.format(layer_i, layer_j)](down_features[layer_j + 1][layer_i - 1])]
                for i in range(layer_i):
                    _cat.append(down_features[layer_j][i])
                _cat = torch.cat(_cat, 1)
                down_features[layer_j].append(self.up_sample_convs['up{}_{}'.format(layer_i, layer_j)](_cat))
        logits = []
        for layer_i in range(1, self.layers):
            logits.append(self.predict_layer['predict{}'.format(layer_i)](down_features[0][layer_i]))
        return logits


    def get_conv_block(self, in_feature, out_feature, prefix):
        _return = OrderedDict()
        for i in range(self.conv_repeat):
            _return[prefix+'_conv{}'.format(i)] = nn.Conv2d(in_feature, out_feature, kernel_size=3, stride=1, padding=1)
            in_feature = out_feature
            if self.use_bn == True:
                _return[prefix+'_norm{}'.format(i)] = nn.BatchNorm2d(out_feature, momentum=self.bn_momentum, track_running_stats=self.track_running_stats)
            elif self.use_gn == True:
                _return[prefix+'_norm{}'.format(i)] = nn.GroupNorm(num_groups=self.num_groups, num_channels=out_feature)
            _return[prefix + '_relu{}'.format(i)] = nn.ReLU(inplace=True)
        return _return


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