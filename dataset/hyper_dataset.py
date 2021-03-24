
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

class hyper_dataset(Dataset):
    """
        下载数据、初始化数据，都可以在这里完成
    """

    def __init__(self, npy_dir, label_dir, split_file, norm_kwargs, channel_transform, aug_config):

        self.norm_kwargs = norm_kwargs
        self.channel_transform = channel_transform
        self.x = []
        self.y = []
        with open(split_file) as f:
            for line in f.readlines():
                self.x.append(os.path.join(npy_dir, line.strip()))
                self.y.append(os.path.join(label_dir, line.strip().replace('0.npy', '_mask.png')))
        self.len = len(self.x)
        # self.mean = [8895.56570294, 8736.16861954, 8792.2975563,  8842.36984379, 8858.12202044,
        #              8808.25820789, 8787.23233831, 8646.04916632, 8505.59169405, 8449.97676027,
        #              8352.90772121, 8288.12660935, 8235.19944885, 8206.08040836, 8202.51871333,
        #              8124.53381603, 8056.29416361, 8049.04513984, 8024.87910366, 8069.70047092,
        #              8118.65904145, 8258.31762395, 8395.65183754, 8540.17889628, 8704.82359512,
        #              8847.05562568, 9007.94572842, 9163.95838697, 9283.36129413, 9426.11386058,
        #              9473.34954653, 9572.36132314, 9637.34273088, 9695.07574816, 9713.16241500,
        #              9746.11757693, 9756.76938274, 9757.60773621, 9790.94245367, 9805.40148313,
        #              9818.19510539, 9833.19961026, 9735.37237427, 9735.13466976, 9772.06312287,
        #              9789.35804327, 9749.81809092, 9782.97123978, 9760.45004332, 9746.35158811,
        #              9734.11583223, 9697.48742757, 9715.78060811, 9724.17147768, 9741.63524325,
        #              9742.35142055, 9737.94356954, 9738.29204242, 9740.23280778, 9740.72090464,]
        # self.std  = [ 899.68596820,  747.64822026,  747.05905316,  750.69992287,  766.99374054,
        #               802.98141416,  878.47214067,  952.15376550, 1047.54590009, 1133.72625741,
        #              1194.16360578, 1255.64688385, 1282.06321218, 1314.63582630, 1365.52798955,
        #              1406.67349841, 1482.59914576, 1543.88792858, 1571.60841381, 1607.24414066,
        #              1595.88683983, 1568.92768374, 1524.55115553, 1486.76795384, 1440.10475697,
        #              1393.82252234, 1352.78132776, 1304.81397990, 1260.02645251, 1202.68307908,
        #              1130.17798827, 1092.39366294, 1041.44663672, 1007.05875218,  981.10384181,
        #               959.72162368,  945.88402096,  926.76589133,  916.27554561,  899.20456073,
        #               863.74476741,  818.49837209,  757.72798220,  712.43105280,  679.83224659,
        #               653.80952062,  633.77023036,  621.68357242,  602.49513177,  588.71498199,
        #               580.31492186,  561.94518793,  553.67218528,  542.58337898,  535.91513926,
        #               531.86331819,  528.98469268,  527.88860393,  526.52153197,  524.72616590, ]
        self.mean = [8767.74507753, 8704.33528893, 8764.76146787, 8823.74236313, 8859.69514695,
                     8837.16513246, 8792.0379876 , 8704.95183511, 8533.700792  , 8480.12577875,
                     8407.32103156, 8318.1674773 , 8290.49329004, 8251.34573434, 8240.06533973,
                     8199.50589202, 8093.38762291, 8112.23262798, 8129.81115241, 8132.94929126,
                     8230.86552891, 8354.09237416, 8504.05907384, 8662.11308906, 8818.7432458,
                     8983.27861134, 9155.95266069, 9292.61352853, 9422.36312335, 9555.92253692,
                     9640.53238117, 9694.76821996, 9795.0561307 , 9819.82870302, 9847.22004009,
                     9876.4136438 , 9905.60690625, 9896.89131326, 9926.73061522, 9924.84012027,
                     9953.75414584, 9919.14214838, 9855.66764303, 9853.46132631, 9827.25613361,
                     9885.0022777 , 9833.54688764, 9820.1838158 , 9828.67573733, 9787.00645107,
                     9791.73297982, 9748.44445561, 9737.40782204, 9765.91671089, 9764.43367338,
                     9756.60531767, 9762.92881418, 9758.20744925, 9760.1832485 , 9761.23269676],
        self.std =  [ 786.40690148,  646.46209235,  644.47730128,  646.71471702,  665.09259162,
                      709.65204137,  788.234844  ,  880.64193853,  975.27485053, 1068.31559186,
                     1137.6271906 , 1192.3429918 , 1226.22350158, 1255.65266396, 1307.78852952,
                     1364.20851072, 1429.02842811, 1496.99072172, 1540.5701182 , 1562.97189646,
                     1559.87362553, 1531.76255772, 1488.65268418, 1448.0766883 , 1405.53092211,
                     1361.08308119, 1321.32763201, 1275.8323443 , 1222.28518207, 1164.17655183,
                     1089.87472044, 1037.55694709,  989.79810197,  944.67915937,  916.67531173,
                      893.8713549 ,  877.89463819,  855.60262008,  842.5042363 ,  817.46170164,
                      778.36017188,  718.42901102,  652.82189621,  596.81521143,  545.53864585,
                      516.38844564,  490.72142761,  468.79087104,  446.6625361 ,  426.24756705,
                      414.9572136 ,  390.64571949,  376.65323326,  360.64912455,  349.05100564,
                      342.1106682 ,  338.48868696,  336.2702614 ,  334.04175279,  331.07240439],
        self.mean = np.array(self.mean, dtype=np.float32).reshape((60, 1, 1))
        self.std = np.array(self.std, dtype=np.float32).reshape((60, 1, 1))
        self.aug_config = aug_config if 'train' in split_file else '10086'


    def __getitem__(self, index):
        data = np.load(self.x[index])
        label = np.array(Image.open(self.y[index]))
        data = self.preprocess(data, index)
        label = np.where(label > 127, True, False)
        label = np.stack([~label, label], 0).astype(np.float32)
        data, label = self.aug(data, label)
        return np.ascontiguousarray(data), np.ascontiguousarray(label)


    def __len__(self):
        return self.len

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
            data = data - np.min(data, axis=(1, 2)).reshape((_C, 1, 1))
            data = data / np.maximum(np.max(data, axis=(1, 2)) / 255, 1).reshape((_C, 1, 1))
        # if self.norm_kwargs['type'] == 'pixel':
        #     data = data - np.min(data, axis=(0))
        #     data = data / (np.max(data, axis=(0)) / 255)
        if self.norm_kwargs['type'] == 'mxt':
            blank = np.load(self.x[index].split('roi')[0]+'blank0.npy')
            data = data / blank

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
            if random.random() < 0.5:
                label = label[:,:,::-1]
                data = data[:,:,::-1]
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
                    label = torch.tensor(np.ascontiguousarray(label)).unsqueeze(0)
                    data = torch.tensor(np.ascontiguousarray(data)).unsqueeze(0)
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






