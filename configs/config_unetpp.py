
from collections import OrderedDict

config_name = 'unetpp'

config_dataset = OrderedDict([
    ('dataset', 'hyper'),
    ('channels', 60),
    ('n_class', 2),
    ('norm_kwargs', {'type':'data'}),
    ('train_dir', '../train/'),
    ('val_dir', '../val/'),
    ('test_dir', '../test/'),
    ('batch_size', 1),
    ('channel_transform', 'none')
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
    ('use_nonlocal', []),
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
    ('workdir', '../workdir/{}_public/'.format(config_name)),
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
config_rawnorm = config_public.copy()
config_rawnorm['norm_kwargs'] = {'type':'raw'}
config_rawnorm['workdir'] = '../workdir/{}_norm_raw_1/'.format(config_name)

config_statnorm = config_public.copy()
config_statnorm['norm_kwargs'] = {'type':'stat'}
config_statnorm['workdir'] = '../workdir/{}_norm_stat_1/'.format(config_name)

config_datanorm = config_public.copy()
config_datanorm['norm_kwargs'] = {'type':'data'}
config_datanorm['workdir'] = '../workdir/{}_norm_data_1/'.format(config_name)
# 
# config_mixednorm = config_public.copy()
# config_mixednorm['norm_kwargs'] = {'type':'stat_data_mixed','rate':[0.5,0.5]}
# config_mixednorm['workdir'] = '../workdir/{}_norm_mixed_2/'.format(config_name)

# config_norm_all = [config_rawnorm, config_statnorm, config_datanorm, config_mixednorm]
# config_norm_all = [config_rawnorm, config_statnorm, config_datanorm]
# config_norm_all = [config_statnorm, config_datanorm]
# config_norm_all = [config_rawnorm, config_statnorm]
# config_norm_all = [config_rawnorm]
config_norm_all = [config_datanorm]
##################################################    norm configs


##################################################    bn configs
config_bn_all = []
for i in range(11):
    config_bn_all.append(config_public.copy())
    config_bn_all[i]['bn_momentum'] = i / 10
    config_bn_all[i]['workdir'] = '../workdir/{}_bn_{}/'.format(config_name, i)
##################################################    bn configs

##################################################    layers configs
config_layer_all = []
for i in range(2, 7):
    config_layer_all.append(config_public.copy())
    config_layer_all[-1]['layers'] = i
    config_layer_all[-1]['workdir'] = '../workdir/{}_layers_{}/'.format(config_name, i)

##################################################    layers configs

##################################################    features configs
config_features_all = []
for i in range(1, 9):
    config_features_all.append(config_public.copy())
    config_features_all[-1]['features_root'] = i * 8
    config_features_all[-1]['workdir'] = '../workdir/{}_features_{}/'.format(config_name, i)

##################################################    features configs


##################################################    features configs
config_nonlocal_all = []
# for i in range(2):
#     config_nonlocal_all.append(config_public.copy())
#     config_nonlocal_all[-1]['use_nonlocal'] = ['down4']
#     config_nonlocal_all[-1]['workdir'] = '../workdir/{}_nonlocal_{}/'.format(config_name, i)
for i in range(2,4):
    config_nonlocal_all.append(config_public.copy())
    config_nonlocal_all[-1]['use_nonlocal'] = ['down4','down3']
    config_nonlocal_all[-1]['workdir'] = '../workdir/{}_nonlocal_{}/'.format(config_name, i)
for i in range(4,6):
    config_nonlocal_all.append(config_public.copy())
    config_nonlocal_all[-1]['use_nonlocal'] = ['up1_3']
    config_nonlocal_all[-1]['workdir'] = '../workdir/{}_nonlocal_{}/'.format(config_name, i)

##################################################    features configs

##################################################    channel_transform configs
config_channel_transform_all = []
for i in range(2):
    config_channel_transform_all.append(config_public.copy())
    config_channel_transform_all[-1]['channel_transform'] = 'none'
    config_channel_transform_all[-1]['workdir'] = '../workdir_old0/{}_channel_transform_{}_none/'.format(config_name, i)
    # config_channel_transform_all[-1]['workdir'] = '../workdir/{}_channel_transform_{}_none/'.format(config_name, i)
# for i in range(11,16):
#     config_channel_transform_all.append(config_public.copy())
#     config_channel_transform_all[-1]['channel_transform'] = 'fft'
#     config_channel_transform_all[-1]['workdir'] = '../workdir/{}_channel_transform_{}_fft/'.format(config_name, i)
# for i in range(21,26):
#     config_channel_transform_all.append(config_public.copy())
#     config_channel_transform_all[-1]['channel_transform'] = 'fake_rgb_10:40:10'
#     config_channel_transform_all[-1]['channels'] = 3
#     config_channel_transform_all[-1]['workdir'] = '../workdir/{}_channel_transform_{}_fake_rgb/'.format(config_name, i)
# for i in range(31,36):
#     config_channel_transform_all.append(config_public.copy())
#     config_channel_transform_all[-1]['channel_transform'] = 'fake_rgb_20'
#     config_channel_transform_all[-1]['channels'] = 1
#     config_channel_transform_all[-1]['workdir'] = '../workdir/{}_channel_transform_{}_fake_rgb/'.format(config_name, i)

##################################################    channel_transform configs

##################################################    weight decay configs
config_weight_decay_all = []
for i in range(1, 9):
    config_weight_decay_all.append(config_public.copy())
    config_weight_decay_all[-1]['weight_decay'] = 10 ** (-i)
    config_weight_decay_all[-1]['workdir'] = '../workdir/{}_weight_decay_{}/'.format(config_name, i)

##################################################    weight decay configs
all_configs = config_weight_decay_all
