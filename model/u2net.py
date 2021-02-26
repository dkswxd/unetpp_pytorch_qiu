import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from collections import OrderedDict
from torch import Tensor
from torch.jit.annotations import List
import numpy as np

class u2net(nn.Module):
    def __init__(self, config):
        super(u2net, self).__init__()
        self.layers = config['layers']
        self.feature_root = config['feature_root']
        self.channels = config['channels']
        self.n_class = config['n_class']
        # self.use_bn = config['use_bn']
        self.track_running_stats = config['track_running_stats']
        # self.bn_momentum = config['bn_momentum']
        # self.use_nonlocal = config['use_nonlocal']
        # self.conv_repeat = config['conv_repeat']
        self.u2net_type = config['u2net_type']

        if config['loss'] == 'BCE':
            self.loss_func = torch.nn.BCELoss()
        else:
            pass

        fr = self.feature_root
        ch = self.channels
        nc = self.n_class
        if self.u2net_type == "full": # dict, in_channel, middle_channel, out_channel

            self.down_sample_convs = torch.nn.ModuleDict()
            # down sample conv layers
            for layer in range(self.layers):
                if layer == 0:
                    self.down_sample_convs['en_{}'.format(layer)] = RSU(
                        self.layers + 1 - layer, ch * 1, fr / 2, fr * 2, self.track_running_stats, 'none')
                elif layer >= self.layers - 2:
                    self.down_sample_convs['en_{}'.format(layer)] = RSUF(
                        4, fr * (2 ** (self.layers - 2)), fr * (2 ** (self.layers - 3)), fr * (2 ** (self.layers - 2)), self.track_running_stats, 'down')
                else:
                    self.down_sample_convs['en_{}'.format(layer)] = RSU(
                        self.layers + 1 - layer, fr * (2 ** layer), fr * (2 ** (layer - 1)), fr * (2 ** (layer + 1)), self.track_running_stats, 'down')

            self.up_sample_convs = torch.nn.ModuleDict()
            # up sample conv layers
            for layer in range(self.layers - 2, -1, -1):
                if layer >= self.layers - 2:
                    self.up_sample_convs['de_{}'.format(layer)] = RSUF(
                        4, fr * (2 ** (self.layers - 1)), fr * (2 ** (self.layers - 3)), fr * (2 ** (self.layers - 2)), self.track_running_stats, 'none')
                else:
                    self.up_sample_convs['de_{}'.format(layer)] = RSU(
                        self.layers + 1 - layer, fr * (2 ** (layer + 2)), fr * (2 ** (layer - 1)), fr * (2 ** layer), self.track_running_stats, 'none')

            self.predict_layer = torch.nn.ModuleDict()
            # predict layers
            for layer in range(self.layers - 1, -1, -1):
                if layer == self.layers - 1:
                    self.predict_layer['sup_{}'.format(layer)] = SUP(layer, fr * (2 ** (layer - 1)), nc) #upsample_times, in_channel, out_channel
                else:
                    self.predict_layer['sup_{}'.format(layer)] = SUP(layer, fr * (2 ** layer), nc) #upsample_times, in_channel, out_channel
            self.predict_layer['sup_fusion'] = self.sup_0 = SUP(0, nc * (self.layers), nc)

        elif self.u2net_type == "small":
            self.down_sample_convs = torch.nn.ModuleDict()
            # down sample conv layers
            for layer in range(self.layers):
                if layer == 0:
                    self.down_sample_convs['en_{}'.format(layer)] = RSU(
                        self.layers + 1 - layer, ch * 1, fr * 1, fr * 2, self.track_running_stats, 'none')
                elif layer >= self.layers - 2:
                    self.down_sample_convs['en_{}'.format(layer)] = RSUF(
                        4, fr * 2, fr / 2, fr * 2, self.track_running_stats, 'down')
                else:
                    self.down_sample_convs['en_{}'.format(layer)] = RSU(
                        self.layers + 1 - layer, fr * 2, fr / 2, fr * 2, self.track_running_stats, 'down')

            self.up_sample_convs = torch.nn.ModuleDict()
            # up sample conv layers
            for layer in range(self.layers - 2, -1, -1):
                if layer >= self.layers - 2:
                    self.up_sample_convs['de_{}'.format(layer)] = RSUF(
                        4, fr * 4, fr / 2, fr * 2, self.track_running_stats, 'none')
                else:
                    self.up_sample_convs['de_{}'.format(layer)] = RSU(
                        self.layers + 1 - layer, fr * 4, fr / 2, fr * 2, self.track_running_stats, 'none')

            self.predict_layer = torch.nn.ModuleDict()
            # predict layers
            for layer in range(self.layers - 1, -1, -1):
                if layer == self.layers - 1:
                    self.predict_layer['sup_{}'.format(layer)] = SUP(layer, fr * 2, nc) #upsample_times, in_channel, out_channel
                else:
                    self.predict_layer['sup_{}'.format(layer)] = SUP(layer, fr * 2, nc) #upsample_times, in_channel, out_channel
            self.predict_layer['sup_fusion'] = self.sup_0 = SUP(0, nc * (self.layers), nc)


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
                down_features.append(self.down_sample_convs['en_{}'.format(layer)](x))
            else:
                down_features.append(self.down_sample_convs['en_{}'.format(layer)](down_features[-1]))

        up_features = []
        for layer in range(self.layers - 2, -1, -1):
            if layer == self.layers - 2:
                _cat = torch.cat((down_features[layer], F.interpolate(down_features[-1], scale_factor=2)), 1)
            else:
                _cat = torch.cat((down_features[layer], F.interpolate(up_features[-1], scale_factor=2)), 1)
            up_features.append(self.up_sample_convs['de_{}'.format(layer)](_cat))

        logits = []
        for layer in range(self.layers):
            if layer == self.layers - 1:
                logits.append(self.predict_layer['sup_{}'.format(layer)](down_features[layer]))
            else:
                logits.append(self.predict_layer['sup_{}'.format(layer)](up_features[- layer - 1]))
        logits.append(self.predict_layer['sup_fusion'](torch.cat(logits, 1)))
        return logits


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
#############################################################################    RSU

class RSU(nn.Module):
    def __init__(self, layers, in_channel, middle_channel, out_channel, track_running_stats, scale_opertaion):
        super(RSU, self).__init__()
        self.layers = layers
        self.track_running_stats = track_running_stats
        middle_channel = int(middle_channel)
        if scale_opertaion == 'down':
            od = OrderedDict([('res_pool', nn.MaxPool2d(kernel_size=2))])
        elif scale_opertaion == 'up':
            od = OrderedDict([('res_upsa', nn.UpsamplingNearest2d(scale_factor=2))])
        else:
            od = OrderedDict()
        od.update(self.get_conv_block(in_channel, out_channel, 'res', 1, 1))
        self.res_w = nn.Sequential(od)

        self.down_sample_convs = torch.nn.ModuleDict()
        # down sample conv layers
        for layer in range(layers):
            if layer == 0:
                self.down_sample_convs['down{}'.format(layer)] = nn.Sequential(
                    self.get_conv_block(out_channel, middle_channel, 'down{}'.format(layer), 1, 1))
            elif layer == layers - 1:
                self.down_sample_convs['down{}'.format(layer)] = nn.Sequential(
                    self.get_conv_block(middle_channel, middle_channel, 'down{}'.format(layer), 2, 2))
            else:
                od = OrderedDict([('down{}_pool'.format(layer), nn.MaxPool2d(kernel_size=2))])
                od.update(self.get_conv_block(middle_channel, middle_channel, 'down{}'.format(layer), 1, 1))
                self.down_sample_convs['down{}'.format(layer)] = nn.Sequential(od)

        self.up_sample_convs = torch.nn.ModuleDict()
        # up sample conv layers
        for layer in range(layers - 2, -1, -1):
            if layer == 0:
                self.up_sample_convs['up{}'.format(layer)] = nn.Sequential(
                    self.get_conv_block(middle_channel * 2, out_channel, 'up{}'.format(layer), 1, 1))
            elif layer == layers - 2:
                self.up_sample_convs['up{}'.format(layer)] = nn.Sequential(
                    self.get_conv_block(middle_channel, middle_channel, 'up{}'.format(layer), 1, 1))
            else:
                self.up_sample_convs['up{}'.format(layer)] = nn.Sequential(
                    self.get_conv_block(middle_channel * 2, middle_channel, 'up{}'.format(layer), 1, 1))

        self.up_sample_transpose = torch.nn.ModuleDict()
        for layer in range(layers - 1, -1, -1):
            self.up_sample_transpose['up{}_transpose'.format(layer)] = nn.UpsamplingNearest2d(scale_factor=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        down_features = [self.res_w(x)]
        for layer in range(self.layers):
            down_features.append(self.down_sample_convs['down{}'.format(layer)](down_features[-1]))

        up_features = [self.up_sample_convs['up{}'.format(self.layers - 2)](down_features[-1])]
        for layer in range(self.layers - 3, -1, -1):
            _cat = torch.cat((down_features[layer + 1], self.up_sample_transpose['up{}_transpose'.format(layer)](up_features[-1])), 1)
            up_features.append(self.up_sample_convs['up{}'.format(layer)](_cat))
        result = up_features[-1] + down_features[0]
        return result

    def get_conv_block(self, in_feature, out_feature, prefix, dilation, padding):
        _return = OrderedDict([
                    (prefix+'_conv', nn.Conv2d(in_feature, out_feature, kernel_size=3, stride=1, dilation=dilation, padding=padding)),
                    (prefix+'_norm', nn.BatchNorm2d(out_feature, track_running_stats=self.track_running_stats)),
                    (prefix+'_relu', nn.ReLU(inplace=True)),
                    ])
        return _return


#############################################################################    RSUF

class RSUF(nn.Module):
    def __init__(self, layers, in_channel, middle_channel, out_channel, track_running_stats, scale_opertaion):
        super(RSUF, self).__init__()
        self.layers = layers
        self.in_channel = in_channel
        self.middle_channel = middle_channel
        self.out_channel = out_channel
        self.track_running_stats = track_running_stats
        middle_channel = int(middle_channel)
        if scale_opertaion == 'down':
            od = OrderedDict([('res_pool', nn.MaxPool2d(kernel_size=2))])
        elif scale_opertaion == 'up':
            od = OrderedDict([('res_upsa', nn.UpsamplingNearest2d(scale_factor=2))])
        else:
            od = OrderedDict()
        od.update(self.get_conv_block(in_channel, out_channel, 'res', 1, 1))
        self.res_w = nn.Sequential(od)

        self.down_sample_convs = torch.nn.ModuleDict()
        # down sample conv layers
        for layer in range(layers):
            if layer == 0:
                self.down_sample_convs['down{}'.format(layer)] = nn.Sequential(
                    self.get_conv_block(out_channel, middle_channel, 'down{}'.format(layer), 1, 1))
            else:
                self.down_sample_convs['down{}'.format(layer)] = nn.Sequential(
                    self.get_conv_block(middle_channel, middle_channel, 'down{}'.format(layer), 2 ** layer, 2 ** layer))

        self.up_sample_convs = torch.nn.ModuleDict()
        # up sample conv layers
        for layer in range(layers - 2, -1, -1):
            if layer == 0:
                self.up_sample_convs['up{}'.format(layer)] = nn.Sequential(
                    self.get_conv_block(middle_channel * 2, out_channel, 'up{}'.format(layer), 1, 1))
            elif layer == layers - 2:
                self.up_sample_convs['up{}'.format(layer)] = nn.Sequential(
                    self.get_conv_block(middle_channel, middle_channel, 'up{}'.format(layer), 1, 1))
            else:
                self.up_sample_convs['up{}'.format(layer)] = nn.Sequential(
                    self.get_conv_block(middle_channel * 2, middle_channel, 'up{}'.format(layer), 2 ** layer, 2 ** layer))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        down_features = [self.res_w(x)]
        for layer in range(self.layers):
            down_features.append(self.down_sample_convs['down{}'.format(layer)](down_features[-1]))

        up_features = [self.up_sample_convs['up{}'.format(self.layers - 2)](down_features[-1])]
        for layer in range(self.layers - 3, -1, -1):
            _cat = torch.cat((down_features[layer + 1], up_features[-1]),1)
            up_features.append(self.up_sample_convs['up{}'.format(layer)](_cat))
        result = up_features[-1] + down_features[0]
        return result

    def get_conv_block(self, in_feature, out_feature, prefix, dilation, padding):
        _return = OrderedDict([
            (prefix + '_conv', nn.Conv2d(in_feature, out_feature, kernel_size=3, stride=1, dilation=dilation, padding=padding)),
            (prefix + '_norm', nn.BatchNorm2d(out_feature, track_running_stats=self.track_running_stats)),
            (prefix + '_relu', nn.ReLU(inplace=True)),
        ])
        return _return

#############################################################################    RSUF

class SUP(nn.Module):
    def __init__(self, upsample_times, in_channel, out_channel):
        super(SUP, self).__init__()

        self.predict_layer = nn.Sequential(OrderedDict([
                ('predict_conv', nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1)),
                # ('predict_smax', nn.Sigmoid()),
                ('predict_smax', nn.Softmax2d()),
                ('predict_upsa', nn.UpsamplingNearest2d(scale_factor=2 ** upsample_times))
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
        return self.predict_layer(x)
