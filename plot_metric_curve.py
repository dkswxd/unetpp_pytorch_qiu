import matplotlib.pyplot as plt
import numpy as np
import os
from configs import config_factory


moving_mean = 0.1
for config in config_factory.all_configs:
    for log_file in os.listdir(config['workdir']):
        if log_file.endswith('.txt'):
            lines = open(os.path.join(config['workdir'],log_file)).readlines()
            if len(lines) < 1000:
                continue
            else:
                accuracy = []
                kappa = []
                precision = []
                sensitivity = []
                specificity = []
                auc = []
                train_loss = []
                val_loss = []
                with open(os.path.join(config['workdir'],log_file)) as f:
                    for line in f.readlines():
                        if line.startswith('accuracy'):
                            accuracy.append(float(line.strip().split(' ')[1]))
                        elif line.startswith('kappa'):
                            kappa.append(float(line.strip().split(' ')[1]))
                        elif line.startswith('precision'):
                            precision.append(float(line.strip().split(' ')[1]))
                        elif line.startswith('sensitivity'):
                            sensitivity.append(float(line.strip().split(' ')[1]))
                        elif line.startswith('specificity'):
                            specificity.append(float(line.strip().split(' ')[1]))
                        elif line.startswith('auc'):
                            auc.append(float(line.strip().split(' ')[1]))
                        elif line.startswith('+++'):
                            train_loss.append(float(line.strip().split('loss:')[-1]))
                        elif line.startswith('---'):
                            val_loss.append(float(line.strip().split('loss:')[-1]))


                plt.clf()
                epochs = list(range(config['epoch']))
                plt.plot(epochs, accuracy[:config['epoch']], label='accuracy')
                plt.plot(epochs, kappa[:config['epoch']], label='kappa')
                plt.plot(epochs, precision[:config['epoch']], label='precision')
                plt.plot(epochs, sensitivity[:config['epoch']], label='sensitivity')
                plt.plot(epochs, specificity[:config['epoch']], label='specificity')
                plt.plot(epochs, auc[:config['epoch']], label='auc')
                plt.legend()
                plt.savefig(os.path.join(config['workdir'],log_file).replace('.txt','_val_metric.png'))

                plt.clf()
                pths = list(range(config['save_interval'] - 1, config['epoch'] + config['save_interval'], config['save_interval']))
                plt.plot(pths, accuracy[config['epoch']:], label='accuracy')
                plt.plot(pths, kappa[config['epoch']:], label='kappa')
                plt.plot(pths, precision[config['epoch']:], label='precision')
                plt.plot(pths, sensitivity[config['epoch']:], label='sensitivity')
                plt.plot(pths, specificity[config['epoch']:], label='specificity')
                plt.plot(pths, auc[config['epoch']:], label='auc')
                plt.legend()
                plt.savefig(os.path.join(config['workdir'],log_file).replace('.txt','_test_metric.png'))

                plt.clf()
                for i in range(1, len(train_loss)):
                    train_loss[i] = train_loss[i] * moving_mean + train_loss[i-1] * (1 - moving_mean)
                plt.plot(train_loss, label='train_loss')
                plt.legend()
                plt.savefig(os.path.join(config['workdir'],log_file).replace('.txt','_train.png'))

                plt.clf()
                for i in range(1, len(val_loss)):
                    val_loss[i] = val_loss[i] * moving_mean + val_loss[i-1] * (1 - moving_mean)
                plt.plot(val_loss, label='val_loss')
                plt.legend()
                plt.savefig(os.path.join(config['workdir'],log_file).replace('.txt','_val.png'))
