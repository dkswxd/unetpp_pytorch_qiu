
from collections import OrderedDict

config_name = 'unet'

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
    ('channel_transform', 'none'),
    ('aug_config', 'none')
])

config_model = OrderedDict([
    ('model', config_name),
    ('layers', 4),
    ('feature_root', 32),
    ('conv_repeat', 2),
    ('use_bn', True),
    ('track_running_stats', False),
    # ('track_running_stats', True),
    ('bn_momentum', 0.1),
    ('use_gn', False),
    ('num_groups', 16),
    ('epoch', 50),
    ('save_interval', 5),
    ('restore', False),  # TODO: restore training not implemented!
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


##################################################    norm configs
config_norm_all = []

# config_norm_all.append(config_public.copy())
# config_norm_all[-1]['norm_kwargs'] = {'type':'raw'}
# config_norm_all[-1]['workdir'] = '../cancer/workdir/{}_norm(raw)_1/'.format(config_name)
# config_norm_all.append(config_public.copy())
# config_norm_all[-1]['norm_kwargs'] = {'type':'stat'}
# config_norm_all[-1]['workdir'] = '../cancer/workdir/{}_norm(stat)_1/'.format(config_name)
# config_norm_all.append(config_public.copy())
# config_norm_all[-1]['norm_kwargs'] = {'type':'data'}
# config_norm_all[-1]['workdir'] = '../cancer/workdir/{}_norm(data)_1/'.format(config_name)
#
# config_norm_all.append(config_public.copy())
# config_norm_all[-1]['norm_kwargs'] = {'type':'raw'}
# config_norm_all[-1]['track_running_stats'] = True
# config_norm_all[-1]['workdir'] = '../cancer/workdir/{}_norm(raw)track(true)_1/'.format(config_name)
# config_norm_all.append(config_public.copy())
# config_norm_all[-1]['norm_kwargs'] = {'type':'stat'}
# config_norm_all[-1]['track_running_stats'] = True
# config_norm_all[-1]['workdir'] = '../cancer/workdir/{}_norm(stat)track(true)_1/'.format(config_name)
# config_norm_all.append(config_public.copy())
# config_norm_all[-1]['norm_kwargs'] = {'type':'data'}
# config_norm_all[-1]['track_running_stats'] = True
# config_norm_all[-1]['workdir'] = '../cancer/workdir/{}_norm(data)track(true)_1/'.format(config_name)
#
# config_norm_all.append(config_public.copy())
# config_norm_all[-1]['norm_kwargs'] = {'type':'raw'}
# config_norm_all[-1]['use_bn'] = False
# config_norm_all[-1]['workdir'] = '../cancer/workdir/{}_norm(raw)bn(false)_1/'.format(config_name)
# config_norm_all.append(config_public.copy())
# config_norm_all[-1]['norm_kwargs'] = {'type':'stat'}
# config_norm_all[-1]['use_bn'] = False
# config_norm_all[-1]['workdir'] = '../cancer/workdir/{}_norm(stat)bn(false)_1/'.format(config_name)
# config_norm_all.append(config_public.copy())
# config_norm_all[-1]['norm_kwargs'] = {'type':'data'}
# config_norm_all[-1]['use_bn'] = False
# config_norm_all[-1]['workdir'] = '../cancer/workdir/{}_norm(data)bn(false)_1/'.format(config_name)

config_norm_all.append(config_public.copy())
config_norm_all[-1]['norm_kwargs'] = {'type':'pixel'}
config_norm_all[-1]['workdir'] = '../cancer/workdir/{}_norm(pixel)_1/'.format(config_name)

##################################################    norm configs


##################################################    bn configs
config_bn_all = []
for i in range(11):
    config_bn_all.append(config_public.copy())
    config_bn_all[i]['bn_momentum'] = i / 10
    config_bn_all[i]['workdir'] = '../cancer/workdir/{}_bn_{}/'.format(config_name, i)
##################################################    bn configs

##################################################    layers configs
config_layer_all = []
for i in range(2, 7):
    config_layer_all.append(config_public.copy())
    config_layer_all[-1]['layers'] = i
    config_layer_all[-1]['workdir'] = '../cancer/workdir/{}_layers_{}/'.format(config_name, i)

##################################################    layers configs

##################################################    features configs
config_features_all = []
for i in range(1, 9):
    config_features_all.append(config_public.copy())
    config_features_all[-1]['features_root'] = i * 8
    config_features_all[-1]['workdir'] = '../cancer/workdir/{}_features_{}/'.format(config_name, i)

##################################################    features configs


##################################################    features configs
config_nonlocal_all = []
# for i in range(2):
#     config_nonlocal_all.append(config_public.copy())
#     config_nonlocal_all[-1]['use_nonlocal'] = ['down4']
#     config_nonlocal_all[-1]['workdir'] = '../cancer/workdir/{}_nonlocal_{}/'.format(config_name, i)
for i in range(2,4):
    config_nonlocal_all.append(config_public.copy())
    config_nonlocal_all[-1]['use_nonlocal'] = ['down4','down3']
    config_nonlocal_all[-1]['workdir'] = '../cancer/workdir/{}_nonlocal_{}/'.format(config_name, i)
for i in range(4,6):
    config_nonlocal_all.append(config_public.copy())
    config_nonlocal_all[-1]['use_nonlocal'] = ['up1_3']
    config_nonlocal_all[-1]['workdir'] = '../cancer/workdir/{}_nonlocal_{}/'.format(config_name, i)

##################################################    features configs

##################################################    channel_transform configs
config_channel_transform_all = []
for i in range(1,6):
    config_channel_transform_all.append(config_public.copy())
    config_channel_transform_all[-1]['channel_transform'] = 'none'
    config_channel_transform_all[-1]['workdir'] = '../cancer/workdir/{}_channel_transform_{}_none/'.format(config_name, i)
    # config_channel_transform_all[-1]['workdir'] = '../cancer/workdir_old0/{}_channel_transform_{}_none/'.format(config_name, i)
# for i in range(11,13):
#     config_channel_transform_all.append(config_public.copy())
#     config_channel_transform_all[-1]['channel_transform'] = 'fft'
#     config_channel_transform_all[-1]['workdir'] = '../cancer/workdir/{}_channel_transform_{}_fft/'.format(config_name, i)
for i in range(21,26):
    config_channel_transform_all.append(config_public.copy())
    config_channel_transform_all[-1]['channel_transform'] = 'fake_rgb_10:40:10'
    config_channel_transform_all[-1]['channels'] = 3
    config_channel_transform_all[-1]['workdir'] = '../cancer/workdir/{}_channel_transform_{}_fake_rgb/'.format(config_name, i)
for i in range(31,36):
    config_channel_transform_all.append(config_public.copy())
    config_channel_transform_all[-1]['channel_transform'] = 'fake_rgb_20'
    config_channel_transform_all[-1]['channels'] = 1
    config_channel_transform_all[-1]['workdir'] = '../cancer/workdir/{}_channel_transform_{}_fake_rgb/'.format(config_name, i)
for i in range(41,46):
    config_channel_transform_all.append(config_public.copy())
    config_channel_transform_all[-1]['channel_transform'] = '0:30'
    config_channel_transform_all[-1]['channels'] = 30
    config_channel_transform_all[-1]['workdir'] = '../cancer/workdir/{}_channel_transform_{}_fake_rgb/'.format(config_name, i)
for i in range(51,56):
    config_channel_transform_all.append(config_public.copy())
    config_channel_transform_all[-1]['channel_transform'] = '10:20'
    config_channel_transform_all[-1]['channels'] = 10
    config_channel_transform_all[-1]['workdir'] = '../cancer/workdir/{}_channel_transform_{}_fake_rgb/'.format(config_name, i)

##################################################    channel_transform configs

##################################################    weight decay configs
# config_weight_decay_all = []
# for i in range(1, 9):
#     config_weight_decay_all.append(config_public.copy())
#     config_weight_decay_all[-1]['weight_decay'] = 10 ** (-i)
#     config_weight_decay_all[-1]['workdir'] = '../cancer/workdir/{}_weight_decay_{}/'.format(config_name, i)
# for i in range(2, 4):
#     config_weight_decay_all.append(config_public.copy())
#     config_weight_decay_all[-1]['weight_decay'] = 10 ** (-i)
#     config_weight_decay_all[-1]['workdir'] = '../cancer/workdir/{}_weight_decay2_{}/'.format(config_name, i)
##################################################    weight decay configs

##################################################    group norm configs
config_group_norm_all = []
for i in range(4):
    config_group_norm_all.append(config_public.copy())
    config_group_norm_all[-1]['use_bn'] = False
    config_group_norm_all[-1]['use_gn'] = True
    config_group_norm_all[-1]['num_groups'] = 8 * (2 ** i)
    config_group_norm_all[-1]['workdir'] = '../cancer/workdir/{}_group_norm_{}/'.format(config_name, 8 * (2 ** i))
##################################################    group norm configs

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
config_focal_loss_all = []
for i in range(5):
    config_focal_loss_all.append(config_public.copy())
    # config_focal_loss_all[-1]['loss'] = 'focal'
    config_focal_loss_all[-1]['train_split'] = '../cancer/split/split_{}_train.txt'.format(i)
    config_focal_loss_all[-1]['val_split'] = '../cancer/split/split_{}_val.txt'.format(i)
    config_focal_loss_all[-1]['test_split'] = '../cancer/split/split_{}_test.txt'.format(i)
    config_focal_loss_all[-1]['workdir'] = '../cancer/workdir/{}_focal_{}/'.format(config_name, i)
##################################################    split configs

##################################################    split configs
config_aug_all = []
for i in range(5):
    config_aug_all.append(config_public.copy())
    config_aug_all[-1]['aug_config'] = 'flip-transpose-rescale0.5'
    config_aug_all[-1]['train_split'] = '../cancer/split/split_{}_train.txt'.format(i)
    config_aug_all[-1]['val_split'] = '../cancer/split/split_{}_val.txt'.format(i)
    config_aug_all[-1]['test_split'] = '../cancer/split/split_{}_test.txt'.format(i)
    config_aug_all[-1]['workdir'] = '../cancer/workdir/{}_aug2_{}/'.format(config_name, i)
##################################################    split configs


##################################################    split configs
config_mxt_all = []
for i in range(5):
    config_mxt_all.append(config_public.copy())
    config_mxt_all[-1]['norm_kwargs'] = {'type':'mxt'}
    config_mxt_all[-1]['train_split'] = '../cancer/split/split_{}_train.txt'.format(i)
    config_mxt_all[-1]['val_split'] = '../cancer/split/split_{}_val.txt'.format(i)
    config_mxt_all[-1]['test_split'] = '../cancer/split/split_{}_test.txt'.format(i)
    config_mxt_all[-1]['workdir'] = '../cancer/workdir/{}_aug2_{}/'.format(config_name, i)
##################################################    split configs
all_configs = config_split_all
# all_configs = []
