from torch import nn
import torch
from .ext.deform import deform_conv
from collections import OrderedDict


class ULSTM(nn.Module):
    def __init__(self, layers, input_channel, num_filter, conv_repeat, use_deform):
        super().__init__()
        self.ublock = _UBlock(layers, input_channel + num_filter, num_filter, conv_repeat, use_deform)

        self._input_channel = input_channel
        self._num_filter = num_filter

    # inputs and states should not be all none
    # inputs: S*B*C*H*W
    def forward(self, inputs=None, states=None, seq_len=None):
        _, _, _, self._state_height, self._state_width = inputs.shape

        if states is None:
            c = torch.zeros((inputs.size(1), self._num_filter, self._state_height,
                                  self._state_width), dtype=torch.float).cuda()
            h = torch.zeros((inputs.size(1), self._num_filter, self._state_height,
                             self._state_width), dtype=torch.float).cuda()
        else:
            h, c = states

        outputs = []
        for index in range(seq_len):
            # initial inputs
            if inputs is None:
                x = torch.zeros((h.size(0), self._input_channel, self._state_height,
                                      self._state_width), dtype=torch.float).cuda()
            else:
                x = inputs[index, ...]
            cat_x = torch.cat([x, h], dim=1)
            conv_x = self.ublock(cat_x)

            i, f, tmp_c, o = torch.chunk(conv_x, 4, dim=1)

            i = torch.sigmoid(i)
            # i = torch.sigmoid(i+self.Wci*c)
            f = torch.sigmoid(f)
            # f = torch.sigmoid(f+self.Wcf*c)
            c = f*c + i*torch.tanh(tmp_c)
            o = torch.sigmoid(o)
            # o = torch.sigmoid(o+self.Wco*c)
            h = o*torch.tanh(c)
            outputs.append(h)
        return torch.stack(outputs), h, c
        # return h, c



class _UBlock(nn.Module):
    def __init__(self, layers, input_channel, num_filter, conv_repeat, use_deform):
        super(_UBlock, self).__init__()
        self.layers = layers
        self.channels = input_channel
        self.feature_root = num_filter
        self.conv_repeat = conv_repeat
        self.use_deform = use_deform

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

        self.predict_layer = nn.Conv2d(self.feature_root, self.feature_root*4, kernel_size=3, stride=1, padding=1)

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
        return self.predict_layer(up_features[-1])


    def get_conv_block(self, in_feature, out_feature, prefix):
        _return = OrderedDict()
        for i in range(self.conv_repeat):
            if prefix+'_conv{}'.format(i) in self.use_deform:
                _return[prefix + '_conv{}'.format(i)] = deform_conv.DeformConv2dPack(in_feature, out_feature, kernel_size=3, stride=1, padding=1)
            else:
                _return[prefix+'_conv{}'.format(i)] = nn.Conv2d(in_feature, out_feature, kernel_size=3, stride=1, padding=1)
            in_feature = out_feature
            _return[prefix+'_norm{}'.format(i)] = nn.BatchNorm2d(out_feature, track_running_stats=False)
            if prefix != 'up0':
                _return[prefix + '_relu{}'.format(i)] = nn.ReLU(inplace=True)
        return _return