from .unet import unet
from .unetpp import unetpp
from .unet_deform import unet_deform
from .unet_3d import unet_3d
from .unet_poi import unet_poi
from .u2net import u2net
from .pspnet import PSPNet
from .BiCLSTM import BiCLSTM
from .BiCLSTM_unet import BiCLSTM_unet
from .FCLSTM_unet import FCLSTM_unet
from .BiCLSTM_dunet import BiCLSTM_dunet
from .BiCLSTM_in_unet import BiCLSTM_in_unet
from .BiULSTM import BiULSTM
from .BiULSTM_seperate import BiULSTM_seperate
from .FULSTM import FULSTM
from .BiULSTM_unet import BiULSTM_unet
from .unet_DS import unet_DS
from .unet_softpool import unet_softpool

model_dict = {'unet': unet,
              'unetpp': unetpp,
              'unet_deform': unet_deform,
              'u2net': u2net,
              'pspnet': PSPNet,
              'unet_poi': unet_poi,
              'unet_3d': unet_3d,
              'BiCLSTM': BiCLSTM,
              'BiCLSTM_unet': BiCLSTM_unet,
              'FCLSTM_unet': FCLSTM_unet,
              'BiCLSTM_dunet': BiCLSTM_dunet,
              'BiCLSTM_in_unet':BiCLSTM_in_unet,
              'BiULSTM': BiULSTM,
              'BiULSTM_seperate': BiULSTM_seperate,
              'FULSTM': FULSTM,
              'BiULSTM_unet': BiULSTM_unet,
              'unet_DS':unet_DS,
              'unet_softpool':unet_softpool,}

def get_model(config):
    assert config['model'] in model_dict.keys()
    return model_dict[config['model']](config).cuda()