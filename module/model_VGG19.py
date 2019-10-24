import torch
import torchvision
import torch.nn.functional as F

from module.sequential import Sequential
from module.linear import Linear
from module.relu import ReLU
from module.convolution import _ConvNd
from module.pool import _MaxPoolNd
from module.module import Module
from module.convolution import Conv2d
from module.batchnorm import BatchNorm2d
from module.pool import MaxPool2d
from module.arguments import get_args

import os
args = get_args()


# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
class model(Module):
    def forward(self, pretrained=False, pretrained_path=None, batch_norm = args.vgg_bn):
        layers = self.make_layers(cfg['E'])
        
        layers = layers + [Linear(512 * 7 * 7, 4096),
                           ReLU(),
                           Linear(4096, 4096),
                           ReLU(),
                           Linear(4096, 1000, whichScore = args.whichScore, lastLayer=True)]
        
        
        if pretrained == False:
            return Sequential(*layers)
        
        net = Sequential(*layers)
        
        if pretrained_path == None : 
            if batch_norm == True:
                vgg19 = torchvision.models.vgg19_bn(pretrained=True)
            else:
                vgg19 = torchvision.models.vgg19(pretrained=True)
            vgg19_keys = list(vgg19.state_dict().keys())
            net_keys = list(net.state_dict().keys())
            
            for i in range(len(vgg19_keys)):
                try:
                    net.state_dict()[net_keys[i]][:] = vgg19.state_dict()[vgg19_keys[i]][:]
                except:
                    net.state_dict()[net_keys[i]] = vgg19.state_dict()[vgg19_keys[i]]
        else : 
            print(os.getcwd())
            net.load_state_dict(torch.load(pretrained_path))
        
        return net



    def make_layers(self, cfg, batch_norm = args.vgg_bn):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, BatchNorm2d(v), ReLU()]
                else:
                    layers += [conv2d, ReLU()]
                in_channels = v
        return layers


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']}
