from model.model_factory import get_model
from dataset.hyper_dataset import hyper_dataset
from configs import config_factory
from tool import send_email
from tool import metric

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm
import datetime

for config in config_factory.all_configs:
    config_str = '\n***config***\n'
    for k,v in config.items():
        config_str += '{}: {}\n'.format(k, v)
    print(config_str)
    try:
        os.mkdir(config['workdir'])
    except:
        pass
    log = open(os.path.join(config['workdir'], '{}.txt'.format(datetime.datetime.now())), 'w')
    log.write(config_str)

    ### build model
    model = get_model(config)
    model_str = '\n***model***\n{}'.format(model)
    print(model_str)
    log.write(model_str)
    # config_str += model_str

    ## build optimizer
    if config['optimizer'] == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, config['scheduler'], gamma=config['scheduler_rate'])
    elif config['optimizer'] == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'], momentum=0.9)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, config['scheduler'], gamma=config['scheduler_rate'])
    else:
        print('optimizer not implemented!')
        send_email.send_email(config_str, config['optimizer'] + ' optimizer not implemented!')
        continue

    ## build dataset
    if config['dataset'] == 'hyper':
        train_dataset = hyper_dataset(config['npy_dir'], config['label_dir'], config['train_split'], config['norm_kwargs'], config['channel_transform'], config['aug_config'])
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
        val_dataset = hyper_dataset(config['npy_dir'], config['label_dir'], config['val_split'], config['norm_kwargs'], config['channel_transform'], config['aug_config'])
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)
        test_dataset = hyper_dataset(config['npy_dir'], config['label_dir'], config['test_split'], config['norm_kwargs'], config['channel_transform'], config['aug_config'])
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    else:
        print('dataset not implemented!')
        send_email.send_email(config_str, config['dataset'] + ' dataset not implemented!')
        continue

    writer = SummaryWriter(config['workdir'])

    best_val_score = 0

    for epoch in range(config['epoch']):
        # train
        if 'train' in config['work_phase']:
            model.train()
            for step, (batch_x, batch_y) in enumerate(tqdm(train_loader)):
                batch_x = batch_x.cuda()
                batch_y = batch_y.cuda()
                logits = model(batch_x)
                loss = model.get_loss(logits, batch_y)
                with torch.no_grad():
                    # print('epoch:{}, step:{}, train loss:{}'.format(epoch, step, loss))
                    log.write('+++time:{}, epoch:{}, step:{}, train loss:{}\n'.format(datetime.datetime.now(), epoch, step, loss))
                    writer.add_scalar('trainloss', loss.item(), global_step=step + epoch * train_dataset.len)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            scheduler.step()
            if epoch % config['save_interval'] == config['save_interval'] - 1:
                torch.save(model.state_dict(), os.path.join(config['workdir'] ,"epoch_{}.pth".format(epoch)))
            torch.cuda.empty_cache()

        # val
        if 'val' in config['work_phase']:
            with torch.no_grad():
                model.eval()
                metrics_ = []
                for step, (batch_x, batch_y) in enumerate(tqdm(val_loader)):
                    batch_x = batch_x.cuda()
                    batch_y = batch_y.cuda()
                    logits = model(batch_x)
                    loss = model.get_loss(logits, batch_y)
                    # print('epoch:{}, step:{}, train loss:{}'.format(epoch, step, loss))
                    log.write('---time:{}, epoch:{}, step:{}, test loss:{}\n'.format(datetime.datetime.now(), epoch, step, loss))
                    writer.add_scalar('testloss', loss.item(), global_step=step + epoch * test_dataset.len)
                    optimizer.zero_grad()
                    ### metric
                    prediction = model.get_predict(logits)
                    gt = model.get_gt(batch_y)
                    metrics_.append(metric.get_metrics(prediction, logits, gt))
                result_str = '\n***epoch_{}_result***\n'.format(epoch)
                for k,v in metric.show_metrics(metrics_).items():
                    result_str += '{}: {}\n'.format(k, v)
                print(result_str)
                log.write(result_str)
                if metric.show_metrics(metrics_)['kappa'] > best_val_score:
                    torch.save(model.state_dict(), os.path.join(config['workdir'], "best_val_score.pth".format(epoch)))
                    best_val_score = metric.show_metrics(metrics_)['kappa']
                torch.cuda.empty_cache()


    # test

    if 'test' in config['work_phase']:
        with torch.no_grad():
            for epoch in range(config['save_interval'] - 1, config['epoch'] + config['save_interval'], config['save_interval']):
                if epoch <= config['epoch']:
                    test_epoch = "epoch_{}.pth".format(epoch)
                else:
                    test_epoch = "best_val_score.pth"
                model.load_state_dict(torch.load(os.path.join(config['workdir'] ,test_epoch)))
                model.eval()
                # model.train()
                metrics_ = []
                for step, (batch_x, batch_y) in enumerate(tqdm(test_loader)):
                    batch_x = batch_x.cuda()
                    # batch_y = batch_y.cuda()
                    logits = model(batch_x)
                    ### metric
                    prediction = model.get_predict(logits)
                    gt = model.get_gt(batch_y)
                    metrics_.append(metric.get_metrics(prediction, logits, gt))
                result_str = '\n***test_{}_result***\n'.format(test_epoch)
                for k, v in metric.show_metrics(metrics_).items():
                    result_str += '{}: {}\n'.format(k, v)
                print(result_str)
                log.write(result_str)
                config_str += result_str
                torch.cuda.empty_cache()
    send_email.send_email(config_str, config['workdir'] + ' finished!')