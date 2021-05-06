import re
import torch
from torch import nn, einsum
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from collections import OrderedDict
from torch import Tensor
from torch.jit.annotations import List
import numpy as np
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class HSI_Tr(nn.Module):
    def __init__(self, config):
        super(HSI_Tr, self).__init__()
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

        self.preprocessBN = nn.BatchNorm2d(self.channels, track_running_stats=False)

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


        self.to_pixel_embedding = nn.Sequential(
            Rearrange('b c s h w -> (b h w) s c')
        )
        self.transformer = Transformer(self.feature_root, 2, 4, 4, 4)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.channels, self.feature_root),requires_grad=True)
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.feature_root),requires_grad=True)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.feature_root),
            nn.Linear(self.feature_root, self.feature_root)
        )
        self.to_image = Rearrange('b h w c -> b c h w')

        self.predict_layer = nn.Sequential(OrderedDict([
                ('predict_conv', nn.Conv2d(32, self.n_class, kernel_size=3, stride=1, padding=1)),
                # ('predict_smax', nn.Sigmoid()),
                ('predict_smax', nn.Softmax2d()),
                ]))


    def forward(self, x):
        x = self.preprocessBN(x)
        x = x.unsqueeze(1)
        x = F.max_pool3d(x,kernel_size=2)
        # # convert x from 1x32x1024x1280 to 1x1x32x1024x1280


        down_features = []
        for layer in range(self.layers):
            if layer == 0:
                down_features.append(self.down_sample_convs['down{}'.format(layer)](x))
            else:
                x = F.max_pool3d(down_features[-1], kernel_size=(1,2,2))
                down_features.append(self.down_sample_convs['down{}'.format(layer)](x))
        up_features = []
        for layer in range(self.layers - 2, -1, -1):
            if layer == self.layers - 2:
                _cat = torch.cat((down_features[layer], F.interpolate(down_features[layer + 1], scale_factor=(1,2,2))), 1)
            else:
                _cat = torch.cat((down_features[layer], F.interpolate(up_features[-1], scale_factor=(1,2,2))), 1)
            up_features.append(self.up_sample_convs['up{}'.format(layer)](_cat))

        x = up_features[-1]

        x = self.to_pixel_embedding(x)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.transformer(x)
        x = x.reshape((1, 512, 640, -1))
        x = self.to_image(x)
        logits = self.predict_layer(x)
        logits = F.interpolate(logits,scale_factor=2)
        return logits


    def get_conv_block(self, in_feature, out_feature, prefix):
        _return = OrderedDict()
        for i in range(self.conv_repeat):
            _return[prefix+'_conv{}'.format(i)] = nn.Conv3d(in_feature, out_feature, kernel_size=(1,3,3), stride=1, padding=(0,1,1))
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




class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x