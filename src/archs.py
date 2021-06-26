from typing import List

import torch
import torch.nn as nn

from resnet_cifar import resnet32
from vgg import vgg11_nodropout, vgg11_nodropout_bn
from data import num_classes, num_input_channels, image_size, num_pixels

_CONV_OPTIONS = {"kernel_size": 3, "padding": 1, "stride": 1}

def get_activation(activation: str):
    if activation == 'relu':
        return torch.nn.ReLU()
    elif activation == 'hardtanh':
        return torch.nn.Hardtanh()
    elif activation == 'leaky_relu':
        return torch.nn.LeakyReLU()
    elif activation == 'selu':
        return torch.nn.SELU()
    elif activation == 'elu':
        return torch.nn.ELU()
    elif activation == "tanh":
        return torch.nn.Tanh()
    elif activation == "softplus":
        return torch.nn.Softplus()
    elif activation == "sigmoid":
        return torch.nn.Sigmoid()
    else:
        raise NotImplementedError("unknown activation function: {}".format(activation))

def get_pooling(pooling: str):
    if pooling == 'max':
        return torch.nn.MaxPool2d((2, 2))
    elif pooling == 'average':
        return torch.nn.AvgPool2d((2, 2))


def fully_connected_net(dataset_name: str, widths: List[int], activation: str, bias: bool = True) -> nn.Module:
    modules = [nn.Flatten()]
    for l in range(len(widths)):
        prev_width = widths[l - 1] if l > 0 else num_pixels(dataset_name)
        modules.extend([
            nn.Linear(prev_width, widths[l], bias=bias),
            get_activation(activation),
        ])
    modules.append(nn.Linear(widths[-1], num_classes(dataset_name), bias=bias))
    return nn.Sequential(*modules)


def fully_connected_net_bn(dataset_name: str, widths: List[int], activation: str, bias: bool = True) -> nn.Module:
    modules = [nn.Flatten()]
    for l in range(len(widths)):
        prev_width = widths[l - 1] if l > 0 else num_pixels(dataset_name)
        modules.extend([
            nn.Linear(prev_width, widths[l], bias=bias),
            get_activation(activation),
            nn.BatchNorm1d(widths[l])
        ])
    modules.append(nn.Linear(widths[-1], num_classes(dataset_name), bias=bias))
    return nn.Sequential(*modules)


def convnet(dataset_name: str, widths: List[int], activation: str, pooling: str, bias: bool) -> nn.Module:
    modules = []
    size = image_size(dataset_name)
    for l in range(len(widths)):
        prev_width = widths[l - 1] if l > 0 else num_input_channels(dataset_name)
        modules.extend([
            nn.Conv2d(prev_width, widths[l], bias=bias, **_CONV_OPTIONS),
            get_activation(activation),
            get_pooling(pooling),
        ])
        size //= 2
    modules.append(nn.Flatten())
    modules.append(nn.Linear(widths[-1]*size*size, num_classes(dataset_name)))
    return nn.Sequential(*modules)


def convnet_bn(dataset_name: str, widths: List[int], activation: str, pooling: str, bias: bool) -> nn.Module:
    modules = []
    size = image_size(dataset_name)
    for l in range(len(widths)):
        prev_width = widths[l - 1] if l > 0 else num_input_channels(dataset_name)
        modules.extend([
            nn.Conv2d(prev_width, widths[l], bias=bias, **_CONV_OPTIONS),
            get_activation(activation),
            nn.BatchNorm2d(widths[l]),
            get_pooling(pooling),
        ])
        size //= 2
    modules.append(nn.Flatten())
    modules.append(nn.Linear(widths[-1]*size*size, num_classes(dataset_name)))
    return nn.Sequential(*modules)

def make_deeplinear(L: int, d: int, seed=8):
    torch.manual_seed(seed)
    layers = []
    for l in range(L):
        layer = nn.Linear(d, d, bias=False)
        nn.init.xavier_normal_(layer.weight)
        layers.append(layer)
    network = nn.Sequential(*layers)
    return network.cuda()

def make_one_layer_network(h=10, seed=0, activation='tanh', sigma_w=1.9):
    torch.manual_seed(seed)
    network = nn.Sequential(
        nn.Linear(1, h, bias=True),
        get_activation(activation),
        nn.Linear(h, 1, bias=False),
    )
    nn.init.xavier_normal_(network[0].weight, gain=sigma_w)
    nn.init.zeros_(network[0].bias)
    nn.init.xavier_normal_(network[2].weight)
    return network


def load_architecture(arch_id: str, dataset_name: str) -> nn.Module:
    #  ======   fully-connected networks =======
    if arch_id == 'fc-relu':
        return fully_connected_net(dataset_name, [200, 200], 'relu', bias=True)
    elif arch_id == 'fc-elu':
        return fully_connected_net(dataset_name, [200, 200], 'elu', bias=True)
    elif arch_id == 'fc-tanh':
        return fully_connected_net(dataset_name, [200, 200], 'tanh', bias=True)
    elif arch_id == 'fc-hardtanh':
        return fully_connected_net(dataset_name, [200, 200], 'hardtanh', bias=True)
    elif arch_id == 'fc-softplus':
        return fully_connected_net(dataset_name, [200, 200], 'softplus', bias=True)

    #  ======   convolutional networks =======
    elif arch_id == 'cnn-relu':
        return convnet(dataset_name, [32, 32], activation='relu', pooling='max', bias=True)
    elif arch_id == 'cnn-elu':
        return convnet(dataset_name, [32, 32], activation='elu', pooling='max', bias=True)
    elif arch_id == 'cnn-tanh':
        return convnet(dataset_name, [32, 32], activation='tanh', pooling='max', bias=True)
    elif arch_id == 'cnn-avgpool-relu':
        return convnet(dataset_name, [32, 32], activation='relu', pooling='average', bias=True)
    elif arch_id == 'cnn-avgpool-elu':
        return convnet(dataset_name, [32, 32], activation='elu', pooling='average', bias=True)
    elif arch_id == 'cnn-avgpool-tanh':
        return convnet(dataset_name, [32, 32], activation='tanh', pooling='average', bias=True)

    #  ======   convolutional networks with BN =======
    elif arch_id == 'cnn-bn-relu':
        return convnet_bn(dataset_name, [32, 32], activation='relu', pooling='max', bias=True)
    elif arch_id == 'cnn-bn-elu':
        return convnet_bn(dataset_name, [32, 32], activation='elu', pooling='max', bias=True)
    elif arch_id == 'cnn-bn-tanh':
        return convnet_bn(dataset_name, [32, 32], activation='tanh', pooling='max', bias=True)

    #  ======   real networks on CIFAR-10  =======
    elif arch_id == 'resnet32':
        return resnet32()
    elif arch_id == 'vgg11':
        return vgg11_nodropout()
    elif arch_id == 'vgg11-bn':
        return vgg11_nodropout_bn()

    # ====== additional networks ========
    # elif arch_id == 'transformer':
        # return TransformerModelFixed()
    elif arch_id == 'deeplinear':
        return make_deeplinear(20, 50)
    elif arch_id == 'regression':
        return make_one_layer_network(h=100, activation='tanh')

    # ======= vary depth =======
    elif arch_id == 'fc-tanh-depth1':
        return fully_connected_net(dataset_name, [200], 'tanh', bias=True)
    elif arch_id == 'fc-tanh-depth2':
        return fully_connected_net(dataset_name, [200, 200], 'tanh', bias=True)
    elif arch_id == 'fc-tanh-depth3':
        return fully_connected_net(dataset_name, [200, 200, 200], 'tanh', bias=True)
    elif arch_id == 'fc-tanh-depth4':
        return fully_connected_net(dataset_name, [200, 200, 200, 200], 'tanh', bias=True)
