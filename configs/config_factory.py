
from . import config_unet, config_unetpp, config_u2net, config_unet_deform, config_pspnet, config_unet_poi, \
    config_BiCLSTM, config_BiCLSTM_unet, config_BiCLSTM_in_unet, config_unet_3d, config_BiCLSTM_dunet, \
    config_unet_DS, config_FCLSTM_unet, config_unet_softpool, config_BiULSTM

all_configs = []
#
# all_configs += config_unet.all_configs
# all_configs += config_unetpp.all_configs
# all_configs += config_unet_deform.all_configs
# all_configs += config_unet_poi.all_configs
# all_configs += config_unet_3d.all_configs[2:]
# all_configs += config_pspnet.all_configs
# all_configs += config_u2net.all_configs
# all_configs += config_BiCLSTM.all_configs
# all_configs += config_BiCLSTM_unet.all_configs
# all_configs += config_FCLSTM_unet.all_configs
# all_configs += config_BiCLSTM_dunet.all_configs
# all_configs += config_BiCLSTM_in_unet.all_configs
all_configs += config_BiULSTM.all_configs
# all_configs += config_unet_DS.all_configs
# all_configs += config_unet_softpool.all_configs


# all_configs += config_unet.config_norm_all
# all_configs += config_unet.config_focal_loss_all
# all_configs += config_unet_deform.config_modulated_all
# all_configs += config_unet.config_aug_all
# all_configs += config_unet.config_mxt_all


# #################### apply weight decay
# for config in all_configs:
#     config['weight_decay'] = 0.0001
#     config['epoch'] = 25
#     config['workdir'] = config['workdir'].replace('_split_', '_wd0.001_split_')

# #################### change to blood cell dataset
# for config in all_configs:
#     config['channels'] = 75
#     config['npy_dir'] = config['npy_dir'].replace('cancer', 'cell')
#     config['label_dir'] = config['label_dir'].replace('cancer', 'cell')
#     config['train_split'] = config['train_split'].replace('cancer', 'cell')
#     config['val_split'] = config['val_split'].replace('cancer', 'cell')
#     config['test_split'] = config['test_split'].replace('cancer', 'cell')
#     config['workdir'] = config['workdir'].replace('cancer', 'cell')