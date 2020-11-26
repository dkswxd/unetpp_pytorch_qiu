import os
import numpy as np
import cv2
from configs import config_unet, config_unetpp
from tool import metric

using_config = config_unetpp

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


for config in using_config.all_configs:
    predict_dir = os.path.join(config['workdir'], 'preidct')
    for predict_file in os.listdir(predict_dir):
        predict = cv2.imread(os.path.join(predict_dir, predict_file))
        raw = np.load(os.path.join(config['test_dir'], predict_file.replace('_mask.png', '.npy')))
        h, w = raw.shape[:2]
        raw = raw.astype(np.float)
        raw -= np.min(raw, axis=(0, 1))
        raw /= (np.max(raw, axis=(0, 1)) / 255)
        raw = raw.astype(np.uint8)

        result_str = '\nworkdir:{}\ncurrent:{}\n'.format(config['workdir'],predict_file)
        for k, v in metric.show_metrics_from_save_image(predict[:, -w:, :]).items():
            result_str += '{}: {}\n'.format(k, v)
        print(result_str)

        img2show = np.zeros((h, w*3, 3),dtype=np.uint8)
        img2show[:, w:2*w, :] = predict[:, -w:, :] // 2
        img2show[:, -w:, :] = predict[:, -w:, :]

        key = cv2.waitKey(1)
        while (key != ord('q')):
            img2show[:, :w, 0] = raw[:, :, int(cv2.getTrackbarPos('c0','vis_hyper'))]
            img2show[:, :w, 1] = raw[:, :, int(cv2.getTrackbarPos('c1','vis_hyper'))]
            img2show[:, :w, 2] = raw[:, :, int(cv2.getTrackbarPos('c2','vis_hyper'))]
            img2show[:, w:2*w, 2] = raw[:, :, int(cv2.getTrackbarPos('c0','vis_hyper'))]
            cv2.imshow('vis_hyper', img2show)
            key = cv2.waitKey(10)









