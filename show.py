from model.model_factory import get_model
from dataset.hyper_dataset import hyper_dataset
from configs import config_factory
from tool import metric

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

with torch.no_grad():
    for config in config_factory.all_configs:

        try:
            os.mkdir(os.path.join(config['workdir'], 'logits'))
            os.mkdir(os.path.join(config['workdir'], 'preidct'))
        except:
            pass

        ### build model
        model = get_model(config)

        ## build dataset
        if config['dataset'] == 'hyper':
            test_dataset = hyper_dataset(config['npy_dir'], config['label_dir'], config['test_split'], config['norm_kwargs'], config['channel_transform'], config['aug_config'])
            test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        else:
            print('dataset not implemented!')
            continue

        model.load_state_dict(torch.load(os.path.join(config['workdir'], "best_val_score.pth")))
        model.eval()
        for step, (batch_x, batch_y) in enumerate(tqdm(test_loader)):
            batch_x = batch_x.cuda()
            # batch_y = batch_y.cuda()
            logits = model(batch_x)
            batch_x = batch_x.detach().cpu().numpy()
            prediction = model.get_predict(logits)
            filename = os.path.join(os.path.join(config['workdir'], 'preidct'), test_dataset.y[step].split('/')[-1])
            metric.save_predict(filename, batch_x, batch_y, prediction)
            filename = os.path.join(os.path.join(config['workdir'], 'logits'), test_dataset.y[step].split('/')[-1])
            metric.save_logits(filename, model.get_predict(logits, thresh=False))
