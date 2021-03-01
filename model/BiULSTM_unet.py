import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from collections import OrderedDict
from torch import Tensor
from torch.jit.annotations import List
import numpy as np
from .ULSTM import ULSTM

class BiULSTM_unet(nn.Module):
    def __init__(self, config):
        super(BiULSTM_unet, self).__init__()
        self.layers = config['layers']
        self.feature_root = config['feature_root']
        self.channels = config['channels']
        self.n_class = config['n_class']
        self.use_bn = config['use_bn']
        self.track_running_stats = config['track_running_stats']
        self.conv_repeat = config['conv_repeat']

        self.LSTM_layers = config['LSTM_layers']
        self.LSTM_feature_root = config['LSTM_feature_root']
        self.LSTM_conv_repeat = config['LSTM_conv_repeat']
        self.LSTM_use_deform = config['LSTM_use_deform']


        self.BiCLSTM_feature_root = config['BiCLSTM_feature_root']
        if config['loss'] == 'BCE':
            self.loss_func = torch.nn.BCELoss()
        else:
            pass


        self.forwardLSTM = ULSTM(self.LSTM_layers, self.channels, self.LSTM_feature_root, self.LSTM_conv_repeat, self.LSTM_use_deform)
        self.backwardLSTM = ULSTM(self.LSTM_layers, self.channels, self.LSTM_feature_root, self.LSTM_conv_repeat, self.LSTM_use_deform)


        self.down_sample_convs = torch.nn.ModuleDict()
        # down sample conv layers
        for layer in range(self.layers):
            feature_number = self.feature_root * (2 ** layer)
            if layer == 0:
                self.down_sample_convs['down{}'.format(layer)] = nn.Sequential(
                    self.get_conv_block(self.BiCLSTM_feature_root * 2, feature_number, 'down{}'.format(layer)))
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

        down_features = []
        for layer in range(self.layers):
            if layer == 0:
                down_features.append(self.down_sample_convs['down{}'.format(layer)](_cat))
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
        logits = F.interpolate(logits,scale_factor=2,mode='nearest')
        return logits


    def get_conv_block(self, in_feature, out_feature, prefix):
        _return = OrderedDict()
        for i in range(self.conv_repeat):
            _return[prefix+'_conv{}'.format(i)] = nn.Conv2d(in_feature, out_feature, kernel_size=3, stride=1, padding=1)
            in_feature = out_feature
            if self.use_bn == True:
                _return[prefix+'_norm{}'.format(i)] = nn.BatchNorm2d(out_feature, track_running_stats=self.track_running_stats)
            _return[prefix + '_relu{}'.format(i)] = nn.ReLU(inplace=True)
        return _return

#DKJSB


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