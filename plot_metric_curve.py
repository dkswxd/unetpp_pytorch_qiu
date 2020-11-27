import matplotlib.pyplot as plt
import numpy as np

target_file = '/media/qiu/65F33762C14D581B/zzh/unetpp_pytorch/workdir_old0/unetpp_channel_transform_1_none/2020-11-25 13:16:07.075861.txt'
epoch = []
accuracy = []
kappa = []
precision = []
sensitivity = []
specificity = []
auc = []
with open(target_file) as f:
    for line in f.readlines():
        if line.startswith('***epoch'):
            epoch.append(int(line.split('_')[1]))
        elif line.startswith('accuracy'):
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
        elif line.startswith('***test'):
            break
plt.plot(epoch, accuracy, label='accuracy')
plt.plot(epoch, kappa, label='kappa')
plt.plot(epoch, precision, label='precision')
plt.plot(epoch, sensitivity, label='sensitivity')
plt.plot(epoch, specificity, label='specificity')
plt.plot(epoch, auc, label='auc')
plt.legend()
plt.show()
