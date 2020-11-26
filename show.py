from model.unet import unet
from model.unetpp import unetpp
from model.u2net import u2net
from dataset.hyper_dataset import hyper_dataset
from tool import send_email
from tool import metric
from configs import config_unet, config_unetpp, config_u2net

import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm
import datetime

using_config = config_unetpp

for config in using_config.all_configs:

    try:
        os.mkdir(os.path.join(config['workdir'], 'preidct'))
    except:
        pass

    ### build model
    if config['model'] == 'unet':
        model = unet(config).cuda()
    elif config['model'] == 'unetpp':
        model = unetpp(config).cuda()
    elif config['model'] == 'u2net':
        model = u2net(config).cuda()
    else:
        print('model not implemented!')
        continue

    ## build dataset
    if config['dataset'] == 'hyper':
        test_dataset = hyper_dataset(config['test_dir'], config['norm_kwargs'], config['channel_transform'])
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    else:
        print('dataset not implemented!')
        continue

    model.load_state_dict(torch.load(os.path.join(config['workdir'], "epoch_{}.pth".format(config['epoch'] - 1))))
    model.eval()
    for step, (batch_x, batch_y) in enumerate(tqdm(test_loader)):
        batch_x = batch_x.cuda()
        # batch_y = batch_y.cuda()
        prediction = model(batch_x)
        prediction = prediction[-1].detach().cpu().numpy()
        batch_x = batch_x.detach().cpu().numpy()
        filename = os.path.join(os.path.join(config['workdir'], 'preidct'), test_dataset.y[step])
        metric.save_predict(filename, batch_x, batch_y, prediction)
