import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from collections import OrderedDict
from torch import Tensor
from torch.jit.annotations import List
import numpy as np

class unet_poi(nn.Module):
    def __init__(self, config):
        super(unet_poi, self).__init__()
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

        self.loss_func_poi = torch.nn.MSELoss()

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
                ('predict_smax', nn.Softmax2d()),
                ]))

        self.predict_layer_poi = nn.Sequential(OrderedDict([
                ('predict_poi_linear1', nn.Linear(self.feature_root * (2 ** (self.layers - 1)), 100, bias=False)),
                ('predict_poi_linear2', nn.Linear(100, 1, bias=False)),
                ('predict_poi_sigmoid', nn.Sigmoid()),
                ]))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
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
        logits = self.predict_layer(up_features[-1])
        poi = F.adaptive_avg_pool2d(down_features[-1], (1,1)).view(-1)
        poi = self.predict_layer_poi(poi)
        return [poi, logits]


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
        loss = self.loss_func(logits[-1], batch_y)
        loss += self.loss_func_poi(logits[0], batch_y[:,0,...].mean((1, 2))) * 0.4
        return loss

    def get_predict(self, logits, thresh=True):
        poi = logits[0].detach().cpu().numpy()
        logits = logits[-1].detach().cpu().numpy()
        pred = logits[0, 1, :, :]

        predict_list = pred.reshape(-1).copy()
        predict_list.sort()
        thre = predict_list[int(poi * len(predict_list))]
        if thresh:
            pred = np.where(pred > thre, 1, 0)
        return pred

    def get_gt(self, batch_y):
        batch_y = batch_y.detach().cpu().numpy()
        batch_y = batch_y[0, 1, :, :]
        return np.where(batch_y > 0.5, 1, 0)