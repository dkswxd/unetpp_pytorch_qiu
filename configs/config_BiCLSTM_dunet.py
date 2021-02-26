
from collections import OrderedDict

config_name = 'BiCLSTM_dunet'

config_dataset = OrderedDict([
    ('dataset', 'hyper'),
    ('channels', 60),
    ('n_class', 2),
    ('norm_kwargs', {'type':'data'}),
    ('npy_dir', '../cancer/npy/'),
    ('label_dir', '../cancer/label/'),
    ('train_split', '../cancer/split/split_0_train.txt'),
    ('val_split', '../cancer/split/split_0_val.txt'),
    ('test_split', '../cancer/split/split_0_test.txt'),
    ('batch_size', 1),
    ('channel_transform', '5:45:2'),
    ('aug_config', 'none')
])

config_model = OrderedDict([
    ('model', config_name),
    ('layers', 4),
    ('feature_root', 32),
    ('BiCLSTM_feature_root', 8),
    ('conv_repeat', 2),
    ('use_bn', True),
    ('track_running_stats', False),
    # ('track_running_stats', True),
    ('bn_momentum', 0.1),
    ('use_gn', False),
    ('num_groups', 16),
    ('deform_CLSTM', False),
    ('epoch', 50),
    ('save_interval', 5),
    ('restore', False),  # TODO: restore training not implemented!
    ('modulated', True),
    ('use_deform', ['down3_conv0', 'down3_conv1', 'down2_conv0', 'down2_conv1', 'up2_conv0', 'up2_conv1']),
])

config_optimizer = OrderedDict([
    ('loss', 'BCE'),
    ('optimizer', 'Adam'),
    ('learning_rate', 0.001),
    ('weight_decay', 0.001),
    ('scheduler', [40]),
    ('scheduler_rate', 0.1),
])

config_utils = OrderedDict([
    ('workdir', '../cancer/workdir/{}_public/'.format(config_name)),
    ('work_phase', 'train-val-test'),
    # ('work_phase', 'train-test'),
    # ('work_phase', 'test'),
])

config_public = OrderedDict()
config_public.update(config_dataset)
config_public.update(config_model)
config_public.update(config_optimizer)
config_public.update(config_utils)


##################################################    split configs
config_split_all = []
for i in range(5):
    config_split_all.append(config_public.copy())
    config_split_all[-1]['train_split'] = '../cancer/split/split_{}_train.txt'.format(i)
    config_split_all[-1]['val_split'] = '../cancer/split/split_{}_val.txt'.format(i)
    config_split_all[-1]['test_split'] = '../cancer/split/split_{}_test.txt'.format(i)
    config_split_all[-1]['workdir'] = '../cancer/workdir/{}_split_{}/'.format(config_name, i)
##################################################    split configs

##################################################    split configs
config_split_all = []
for i in range(5):
    config_split_all.append(config_public.copy())
    # config_split_all[-1]['deform_CLSTM'] = True
    config_split_all[-1]['train_split'] = '../cancer/split/split_{}_train.txt'.format(i)
    config_split_all[-1]['val_split'] = '../cancer/split/split_{}_val.txt'.format(i)
    config_split_all[-1]['test_split'] = '../cancer/split/split_{}_test.txt'.format(i)
    config_split_all[-1]['workdir'] = '../cancer/workdir/{}_deform_split_{}/'.format(config_name, i)
##################################################    split configs

all_configs = config_split_all
# all_configs = []
