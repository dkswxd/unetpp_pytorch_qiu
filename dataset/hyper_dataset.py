
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import torchvision
import torch
import cv2
import os
from scipy.fftpack import fft, dct

class hyper_dataset(Dataset):
    """
        下载数据、初始化数据，都可以在这里完成
    """

    def __init__(self, dataset_dir, norm_kwargs, channel_transform):

        self.trans = torchvision.transforms.ToTensor()
        self.norm_kwargs = norm_kwargs
        self.dataset_dir = dataset_dir
        self.channel_transform = channel_transform
        self.x = []
        self.y = []
        for file_name in os.listdir(dataset_dir):
            if file_name.endswith('_mask.png'):
                self.x.append(file_name.replace('_mask.png', '.npy'))
                self.y.append(file_name)
        self.len = len(self.x)
        self.mean = [8895.56570294, 8736.16861954, 8792.2975563,  8842.36984379, 8858.12202044,
                     8808.25820789, 8787.23233831, 8646.04916632, 8505.59169405, 8449.97676027,
                     8352.90772121, 8288.12660935, 8235.19944885, 8206.08040836, 8202.51871333,
                     8124.53381603, 8056.29416361, 8049.04513984, 8024.87910366, 8069.70047092,
                     8118.65904145, 8258.31762395, 8395.65183754, 8540.17889628, 8704.82359512,
                     8847.05562568, 9007.94572842, 9163.95838697, 9283.36129413, 9426.11386058,
                     9473.34954653, 9572.36132314, 9637.34273088, 9695.07574816, 9713.16241500,
                     9746.11757693, 9756.76938274, 9757.60773621, 9790.94245367, 9805.40148313,
                     9818.19510539, 9833.19961026, 9735.37237427, 9735.13466976, 9772.06312287,
                     9789.35804327, 9749.81809092, 9782.97123978, 9760.45004332, 9746.35158811,
                     9734.11583223, 9697.48742757, 9715.78060811, 9724.17147768, 9741.63524325,
                     9742.35142055, 9737.94356954, 9738.29204242, 9740.23280778, 9740.72090464,]
        self.std  = [ 899.68596820,  747.64822026,  747.05905316,  750.69992287,  766.99374054,
                      802.98141416,  878.47214067,  952.15376550, 1047.54590009, 1133.72625741,
                     1194.16360578, 1255.64688385, 1282.06321218, 1314.63582630, 1365.52798955,
                     1406.67349841, 1482.59914576, 1543.88792858, 1571.60841381, 1607.24414066,
                     1595.88683983, 1568.92768374, 1524.55115553, 1486.76795384, 1440.10475697,
                     1393.82252234, 1352.78132776, 1304.81397990, 1260.02645251, 1202.68307908,
                     1130.17798827, 1092.39366294, 1041.44663672, 1007.05875218,  981.10384181,
                      959.72162368,  945.88402096,  926.76589133,  916.27554561,  899.20456073,
                      863.74476741,  818.49837209,  757.72798220,  712.43105280,  679.83224659,
                      653.80952062,  633.77023036,  621.68357242,  602.49513177,  588.71498199,
                      580.31492186,  561.94518793,  553.67218528,  542.58337898,  535.91513926,
                      531.86331819,  528.98469268,  527.88860393,  526.52153197,  524.72616590, ]
        self.mean = np.array(self.mean, dtype=np.float32)
        self.std = np.array(self.std, dtype=np.float32)


    def __getitem__(self, index):
        data = np.load(os.path.join(self.dataset_dir, self.x[index]))
        label = np.array(Image.open(os.path.join(self.dataset_dir, self.y[index])))
        data = self.preprocess(data)
        label = np.where(label > 127, True, False)
        label = np.stack([~label, label], 0).astype(np.float32)
        return self.trans(data), label


    def __len__(self):
        return self.len


    def preprocess(self, data):
        data = data.astype(np.float32)
        if self.norm_kwargs['type'] == 'raw':
            data -= np.amin(data)
            if np.amax(data) != 0:
                data /= np.amax(data)
        if self.norm_kwargs['type'] == 'stat':
            data = data - self.mean
            data = data / self.std
        if self.norm_kwargs['type'] == 'data':
            data = data - data.mean(axis=(0,1))
            data = data / data.std(axis=(0,1))
        if self.norm_kwargs['type'] == 'stat_data_mixed':
            mean = self.mean * self.norm_kwargs['rate'][0] + data.mean(axis=(0,1)) * self.norm_kwargs['rate'][1]
            std = self.std * self.norm_kwargs['rate'][0] + data.std(axis=(0,1)) * self.norm_kwargs['rate'][1]
            data = data - mean
            data = data / std

        if self.channel_transform == 'fft':
            data = dct(data)
        if self.channel_transform == 'fake_rgb_10:40:10':
            data = data[:,:,10:40:10]
        if self.channel_transform == 'fake_rgb_20':
            data = data[:,:,20]
        return data
