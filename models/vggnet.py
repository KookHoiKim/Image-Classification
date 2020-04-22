import torch.nn as nn
import torch.nn.init as init


__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16FCN', 'vgg16FCN_bn', 
    'vgg16', 'vgg16_bn', 'vgg19_bn', 'vgg19'
]



class VGG(nn.Module):

    def __init__(self, conv_layers):

        super(VGG, self).__init__()

        self.conv_module = conv_layers
        
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 1000)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv_module(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'FCN':
            FCN = nn.Conv2d(in_channels, in_channels, kernel_size=1)
            if batch_norm:
                layers += [FCN, nn.BatchNorm2d(in_channels), nn.ReLU()]
            else:
                layers += [FCN, nn.ReLU()]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            FCN = nn.Conv2d(in_channels, in_channels, kernel_size=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU()]
            else:
                layers += [conv2d, nn.ReLU()]
            in_channels = v

    return nn.Sequential(*layers)


cfg = {
    'A' : [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B' : [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'C' : [64, 64, 'M', 128, 128, 'M', 256, 256, 'FCN', 'M', 512, 512, 'FCN', 'M', 512, 512, 'FCN', 'M'],
    'D' : [64, 64,  'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E' : [64, 64,  'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512,
        512, 512, 'M']
}

def vgg11():
    '''VGG 11 without batch normalization'''
    return VGG(make_layers(cfg['A']))

def vgg11_bn():
    """VGG 11 with batch normalization"""
    return VGG(make_layers(cfg['A']), batch_norm=True)

def vgg13():
    return VGG(make_layers(cfg['B']))

def vgg13_bn():
    return VGG(make_layers(cfg['B']), batch_norm=True)

def vgg16FCN():
    '''VGG 16 with Fully Convolutional Network'''
    return VGG(make_layers(cfg['C']))

def vgg16FCN_bn():
    return VGG(make_layers(cfg['C']), batch_norm=True)

def vgg16():
    return VGG(make_layers(cfg['D']))

def vgg16_bn():
    return VGG(make_layers(cfg['D']), batch_norm=True)

def vgg19():
    return VGG(make_layers(cfg['E']))

def vgg19_bn():
    return VGG(make_layers(cfg['E'], batch_norm=True))

