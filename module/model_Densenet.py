import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
import torchvision

from module.sequential import Sequential
from module.linear import Linear
from module.relu import ReLU
from module.convolution import _ConvNd
from module.pool import _MaxPoolNd
from module.module import Module
from module.convolution import Conv2d
from module.pool import MaxPool2d
from module.arguments import get_args
from module.batchnorm import BatchNorm2d
from module.adaptiveAvgPool2d import AdaptiveAvgPool2d, AvgPool2d
from module.arguments import get_args
args = get_args()

__all__ = ['DenseNet', 'densenet121', 'densenet169', 'densenet201', 'densenet161']


model_urls = {
    'densenet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth',
    'densenet169': 'https://download.pytorch.org/models/densenet169-b2777c0a.pth',
    'densenet201': 'https://download.pytorch.org/models/densenet201-c1103571.pth',
    'densenet161': 'https://download.pytorch.org/models/densenet161-8d451a50.pth',
}


class _DenseLayer(Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.layers = []
        self.add_module('1',BatchNorm2d(num_input_features))
        self.add_module('2',ReLU(inplace=False))
        self.add_module('3',Conv2d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False)) # L*k -> 4*k
        self.add_module('4',BatchNorm2d(bn_size * growth_rate))
        self.add_module('5',ReLU(inplace=False))
        self.add_module('6',Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)) # 4*k -> k
        self.drop_rate = drop_rate
        self.growth_rate = growth_rate
        
    def forward(self, x):
        """
        <Chennel dim>
        X : L*k
        out : k (from L*k -> 4*k -> k)
        return : (L+1)*k
        """
        out = x
        for module in self._modules.values():
            out = module(out)
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        return torch.cat([x, out], 1)
    
    def _simple_lrp(self, R, labels):
        """
        <Chennel dim>
        input R : (L+1)*k 
        R_init : L*k
        R0 : k
        R : L*k  (from k -> 4*k -> L*k)
        return : L*k (L*k + L*k)
        """
        
        R_init, R = R[:,:-self.growth_rate,:,:], R[:,-self.growth_rate:,:,:]
        for key, module in enumerate(reversed(self._modules.values())):
            R = module.lrp(R, labels, args.r_method)
            
        return R_init + R
 
    def _composite_lrp(self, R, labels):
        return self._simple_lrp(R, labels)
    
    def _composite_new_lrp(self, R, labels):
        return self._simple_lrp(R, labels)
    
    def _grad_cam(self, dx, requires_activation):
        dx_init, dx = dx[:,:-self.growth_rate,:,:], dx[:,-self.growth_rate:,:,:]
        for key, module in enumerate(reversed(self._modules.values())):
            dx, x = module.grad_cam(dx, requires_activation)
        return dx_init + dx, x


def _DenseBlock(num_layers, num_input_features, bn_size, growth_rate, drop_rate):
    layers = []
    for i in range(num_layers):
        layers.append(_DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate))
        
    return layers
    
class _Transition(Module):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.layers = []
        self.add_module('1',BatchNorm2d(num_input_features))
        self.add_module('2',ReLU(inplace=False))
        self.add_module('3',Conv2d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False))
        self.add_module('4',AvgPool2d(kernel_size=2, stride=2))
    
    def forward(self, x):
        """
        <Channel dim>
        out : about 1/2 of X (because of conv)
        
        <H, W dim>
        out : about 1/4 of X (because of AvgPool2d)
        """
        out = x
        for module in self._modules.values():
            out = module(out)
        return out
    
    def _simple_lrp(self, R, labels):
        for key, module in enumerate(reversed(self._modules.values())):
            R = module.lrp(R, labels, args.r_method)
        return R
 
    def _composite_lrp(self, R, labels):
        return self._simple_lrp(R, labels)
    
    def _composite_new_lrp(self, R, labels):
        return self._simple_lrp(R, labels)
    
    def _grad_cam(self, dx, requires_activation):
        for key, module in enumerate(reversed(self._modules.values())):
            dx, x = module.grad_cam(dx, requires_activation)
        if requires_activation:
            return dx, x
        return dx, None

class DenseNet(Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """

    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000):

        super(DenseNet, self).__init__()

        # First convolution
        self.layers = []
        self.layers.append(Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False))
        self.layers.append(BatchNorm2d(num_init_features))
        self.layers.append(ReLU(inplace=False))
        self.layers.append(MaxPool2d(kernel_size=3, stride=2, padding=1))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.layers += block
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.layers.append(trans)
                num_features = num_features // 2

        # Final batch norm
        self.layers.append(BatchNorm2d(num_features))

        # Linear layer
        self.layers.append(ReLU(inplace=False))
        self.layers.append(AdaptiveAvgPool2d((1, 1)))
        self.layers.append(Linear(num_features, num_classes, whichScore = args.whichScore,lastLayer=True))

    def forward(self):
        return Sequential(*self.layers)


def _load_state_dict(model, model_url):
    # '.'s are no longer allowed in module names, but previous _DenseLayer
    # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
    # They are also in the checkpoints in model_urls. This pattern is used
    # to find such keys.
    model_pretrained = torchvision.models.densenet121(pretrained=True)
    
    for key1, key2 in zip(model.state_dict().keys(), model_pretrained.state_dict().keys()):
        a = model.state_dict()[key1]
        try:
            model.state_dict()[key1][:] = model_pretrained.state_dict()[key2][:]
        except:
            model.state_dict()[key1] = model_pretrained.state_dict()[key2]
            


def densenet121(pretrained=False,pretrained_path=None, **kwargs):
    r"""Densenet-121 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DenseNet(num_init_features=64, growth_rate=32, block_config=([6, 12, 24, 16]), **kwargs).forward()
    if pretrained:
        if pretrained_path == None :
            _load_state_dict(model, model_urls['densenet121'])
        else:
            model.load_state_dict(torch.load(pretrained_path))
    return model


def densenet169(pretrained=False, **kwargs):
    r"""Densenet-169 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 32, 32),
                     **kwargs)
    if pretrained:
        _load_state_dict(model, model_urls['densenet169'])
    return model


def densenet201(pretrained=False, **kwargs):
    r"""Densenet-201 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 48, 32),
                     **kwargs)
    if pretrained:
        _load_state_dict(model, model_urls['densenet201'])
    return model


def densenet161(pretrained=False, **kwargs):
    r"""Densenet-161 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DenseNet(num_init_features=96, growth_rate=48, block_config=(6, 12, 36, 24),
                     **kwargs)
    if pretrained:
        _load_state_dict(model, model_urls['densenet161'])
    return model