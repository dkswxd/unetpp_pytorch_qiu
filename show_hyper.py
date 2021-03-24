import os
import numpy as np
import cv2
from tool import metric

from configs import config_factory


single_channel = False

def callback(object):
    if single_channel == True:
        c0_pos = cv2.getTrackbarPos('c0','vis_hyper')
        cv2.setTrackbarPos('c1','vis_hyper',c0_pos)
        cv2.setTrackbarPos('c2','vis_hyper',c0_pos)
    pass

def single_channel_switch(event, x, y, flags, param):
    global single_channel
    if event == cv2.EVENT_LBUTTONDBLCLK:
        single_channel = not single_channel
        if single_channel == True:
            c0_pos = cv2.getTrackbarPos('c0','vis_hyper')
            cv2.setTrackbarPos('c1','vis_hyper',c0_pos)
            cv2.setTrackbarPos('c2','vis_hyper',c0_pos)

cv2.namedWindow('vis_hyper')
cv2.createTrackbar('c0', 'vis_hyper', 0, 59, callback)
cv2.createTrackbar('c1', 'vis_hyper', 0, 59, callback)
cv2.createTrackbar('c2', 'vis_hyper', 0, 59, callback)
cv2.setMouseCallback('vis_hyper', single_channel_switch)

print('######################################################')
print('left image: fake rgb image using channel:c0, c1, c2')
print('middle image: c0(red), gt(green) and prediction(blue)')
print('right image: gt(green) and prediction(blue)')
print('use q to view next image')
print('######################################################')

all_prediction_file = []
npy_dir = config_factory.all_configs[0]['npy_dir']
for config in config_factory.all_configs:
    predict_dir = os.path.join(config['workdir'], 'preidct')
    for predict_file in os.listdir(predict_dir):
        all_prediction_file.append((predict_dir, predict_file))

i = 0
key = cv2.waitKey(1)
while (key != ord('q')):
        if key == ord('a'):
            i -= 1
        elif key == ord('d'):
            i += 1


        prediction_file = all_prediction_file[i]
        predict = cv2.imread(os.path.join(prediction_file[0], prediction_file[1]))
        raw = np.load(os.path.join(npy_dir, prediction_file[1].replace('_mask.png', '0.npy')))
        h, w = raw.shape[1:]
        raw = raw.astype(np.float)
        raw = raw - np.min(raw, axis=(1, 2)).reshape((60, 1, 1))
        raw = raw / (np.max(raw, axis=(1, 2)) / 255).reshape((60, 1, 1))
        # raw = raw - np.min(raw)
        # raw = raw / (np.max(raw) / 255)
        raw = raw.astype(np.uint8)

        result_str = '\nworkdir:{}\ncurrent:{}\n'.format(prediction_file[0],prediction_file[1])
        for k, v in metric.show_metrics_from_save_image(predict[:, -w:, :]).items():
            result_str += '{}: {}\n'.format(k, v)
        print(result_str)

        img2show = np.zeros((h, w*3, 3),dtype=np.uint8)
        img2show[:, w:2*w, :] = predict[:, -w:, :] // 2
        img2show[:, -w:, :] = predict[:, -w:, :]

        key = cv2.waitKey(10)
        while (key == -1):
            img2show[:, :w, 0] = raw[int(cv2.getTrackbarPos('c0','vis_hyper')),:, :]
            img2show[:, :w, 1] = raw[int(cv2.getTrackbarPos('c1','vis_hyper')),:, :]
            img2show[:, :w, 2] = raw[int(cv2.getTrackbarPos('c2','vis_hyper')),:, :]
            img2show[:, w:2*w, 2] = raw[int(cv2.getTrackbarPos('c0','vis_hyper')),:, :]
            img2show_ = cv2.resize(img2show,(w*3//2, h//2), interpolation=cv2.INTER_NEAREST)
            cv2.imshow('vis_hyper', img2show_)
            key = cv2.waitKey(10)









