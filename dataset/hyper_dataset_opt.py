
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import torchvision
import torch
import cv2
import os
from scipy.fftpack import fft, dct
import random
from skimage.transform import rescale
import torch.nn.functional as F

class hyper_dataset_opt(Dataset):
    """
        下载数据、初始化数据，都可以在这里完成
    """

    def __init__(self, npy_dir, label_dir, split_file, norm_kwargs, channel_transform, aug_config, opt_png_dir='../opt/images/', opt_mask_dir='../opt/masks/'):

        self.norm_kwargs = norm_kwargs
        self.channel_transform = channel_transform
        self.x = []
        self.y = []
        with open(split_file) as f:
            for line in f.readlines():
                self.x.append(os.path.join(npy_dir, line.strip()))
                self.y.append(os.path.join(label_dir, line.strip().replace('.npy', '_mask.png')))
        self.len = len(self.x)
        self.x_opt = []
        self.y_opt = []
        for filename in os.listdir(opt_png_dir):
            self.x_opt.append(os.path.join(opt_png_dir, filename))
            self.y_opt.append(os.path.join(opt_mask_dir, filename))
        self.opt_list = []
        self.aug_config = aug_config if 'train' in split_file else '10086'


    def __getitem__(self, index):
        data = np.load(self.x[index])
        label = np.array(Image.open(self.y[index]))
        data = self.preprocess(data, index)
        label = np.where(label > 127, True, False)
        label = np.stack([~label, label], 0).astype(np.float32)
        data, label = self.aug(data, label)
        ########################
        if len(self.opt_list) == 0:
            self.opt_list = list(range(len(self.x_opt)))
            random.shuffle(self.opt_list)
        opt_index = self.opt_list.pop()
        opt_data = np.array(Image.open(self.x_opt[opt_index]))
        opt_data = self.preprocess_for_opt(opt_data)
        opt_label = np.array(Image.open(self.y_opt[opt_index]))
        opt_label = np.where(opt_label > 127, True, False)
        opt_label = np.stack([~opt_label, opt_label], 0).astype(np.float32)
        opt_data, opt_label = self.aug(data, label)
        ########################
        return data, label, opt_data, opt_label


    def __len__(self):
        return self.len

    def preprocess_for_opt(self, data):
        data = data.astype(np.float32)
        _C , _H, _W = data.shape
        data = data - np.mean(data, axis=(1, 2)).reshape((_C, 1, 1))
        data = data / np.maximum(np.std(data, axis=(1, 2)) / 255, 0.0001).reshape((_C, 1, 1))
        return data

    def preprocess(self, data, index):
        data = data.astype(np.float32)
        _C , _H, _W = data.shape
        # if self.norm_kwargs['type'] == 'raw':
        #     data -= np.amin(data)
        #     if np.amax(data) != 0:
        #         data /= np.amax(data)
        # if self.norm_kwargs['type'] == 'stat':
        #     data = data - self.mean
        #     data = data / self.std
        if self.norm_kwargs['type'] == 'data':
            data = data - np.mean(data, axis=(1, 2)).reshape((_C, 1, 1))
            data = data / np.maximum(np.std(data, axis=(1, 2)) / 255, 0.0001).reshape((_C, 1, 1))
        # if self.norm_kwargs['type'] == 'pixel':
        #     data = data - np.min(data, axis=(0))
        #     data = data / (np.max(data, axis=(0)) / 255)
        # if self.norm_kwargs['type'] == 'mxt':
        #     blank = np.load(self.x[index].split('roi')[0]+'blank0.npy')
        #     data = data / blank

        if self.channel_transform == 'fft':
            data = dct(data)
        elif self.channel_transform == 'fake_rgb_10:40:10':
            data = data[10:40:10]
        elif self.channel_transform == 'fake_rgb_20':
            data = data[20]
        elif self.channel_transform == '0:30':
            data = data[0:30]
        elif self.channel_transform == '5:35':
            data = data[5:35]
        elif self.channel_transform == '5:45:2':
            data = data[5:45:2]
        elif self.channel_transform == '5:35:2':
            data = data[5:35:2]
        elif self.channel_transform == '2:34':
            data = data[2:34]

        return data

    def aug(self, data, label):
        if 'flip' in self.aug_config:
            if random.random() < 0.5:
                label = label[:,::-1,:]
                data = data[:,::-1,:]
                label = np.ascontiguousarray(label)
                data = np.ascontiguousarray(data)
            if random.random() < 0.5:
                label = label[:,:,::-1]
                data = data[:,:,::-1]
                label = np.ascontiguousarray(label)
                data = np.ascontiguousarray(data)
        if 'transpose' in self.aug_config:
            if random.random() < 0.5:
                label = label.transpose(0, 2, 1)
                data = data.transpose(0, 2, 1)
        if 'rescale0.5' in self.aug_config:
            if random.random() < 0.5:
                scale_rate = random.uniform(0.5, 1.5)
                _,_h,_w = label.shape
        #         label = rescale(label,(1,scale_rate,scale_rate))
        #         data = rescale(data,(1,scale_rate,scale_rate)) # 10 second per image
                with torch.no_grad():
                    label = torch.tensor(label).unsqueeze(0)
                    data = torch.tensor(data).unsqueeze(0)
                    label = F.interpolate(label, scale_factor=scale_rate)
                    data = F.interpolate(data, scale_factor=scale_rate)
                    label = label.squeeze().detach().numpy()
                    data = data.squeeze().detach().numpy()
                _,__h,__w = label.shape
                if __h < _h: # padding to 1024*1280
                    _h_diff_0 = (_h - __h) // 2
                    _h_diff_1 = (_h - __h) - _h_diff_0
                    _w_diff_0 = (_w - __w) // 2
                    _w_diff_1 = (_w - __w) - _w_diff_0
                    label = np.pad(label, ((0, 0), (_h_diff_0, _h_diff_1), (_w_diff_0, _w_diff_1)), 'constant', constant_values=0)
                    data = np.pad(data, ((0, 0), (_h_diff_0, _h_diff_1), (_w_diff_0, _w_diff_1)), 'constant', constant_values=0)
                else: # crop to 1024*1280
                    _h_start = random.randint(0, __h - _h)
                    _w_start = random.randint(0, __w - _w)
                    label = label[:,_h_start:_h_start + _h, _w_start:_w_start+_w]
                    data = data[:,_h_start:_h_start + _h, _w_start:_w_start+_w]
        return data, label






