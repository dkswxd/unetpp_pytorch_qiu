
from collections import OrderedDict

config_name = 'BiCLSTM'

config_dataset = OrderedDict([
    ('dataset', 'hyper'),
    ('channels', 30),
    ('n_class', 2),
    ('norm_kwargs', {'type':'data'}),
    ('npy_dir', '../cancer/npy/'),
    ('label_dir', '../cancer/label/'),
    ('train_split', '../cancer/split/split_0_train.txt'),
    ('val_split', '../cancer/split/split_0_val.txt'),
    ('test_split', '../cancer/split/split_0_test.txt'),
    ('batch_size', 1),
    ('channel_transform', '5:35'),
    ('aug_config', 'none')
])

config_model = OrderedDict([
    ('model', config_name),
    ('feature_root', 8),
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


##################################################    split configs
config_split_all = []
for i in range(5):
    config_split_all.append(config_public.copy())
    config_split_all[-1]['train_split'] = '../cancer/split/split_{}_train.txt'.format(i)
    config_split_all[-1]['val_split'] = '../cancer/split/split_{}_val.txt'.format(i)
    config_split_all[-1]['test_split'] = '../cancer/split/split_{}_test.txt'.format(i)
    config_split_all[-1]['workdir'] = '../cancer/workdir/{}_split_{}/'.format(config_name, i)
##################################################    split configs

all_configs = config_split_all
# all_configs = []
