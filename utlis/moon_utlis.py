import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))

from model.wrn import *
from model.mobilenet import *


def init_net(args, device='cpu'):
    # nets = {net_i: None for net_i in range(n_parties)}
    if args.dataset == 'cifar10':
        n_classes = 10
    elif args.dataset == 'gsc':
        n_classes = 35
    elif args.dataset == 'cifar100':
        n_classes = 100
    else:
        exit('Error: unrecognised dataset!')

    if args.moon_model == 'wrn_cifar':
        model = WRN_nn_fs_cifar(args.depth_factor, 1, 0.3, num_classes=n_classes)
    elif args.moon_model == 'mobile_cifar':
        model = MobileNetV3_Small_cifar(num_classes=n_classes)
    elif args.moon_model == 'wrn_gsc':
        model = WRN_nn_fs_gsc(args.depth_factor, 1, 0.3, num_classes=n_classes)
    elif args.moon_model == 'mobile_gsc':
        model = MobileNetV3_Small_gsc(num_classes=n_classes)
    else:
        exit('Error: unrecognised model type!')

    model.to(device)

    model_meta_data = []
    layer_type = []
    for (k, v) in model.state_dict().items():
        model_meta_data.append(v.shape)
        layer_type.append(k)

    return model, model_meta_data, layer_type
