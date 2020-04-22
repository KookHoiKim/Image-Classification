import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class GoogLeNet(nn.Module):
    #__constants__ = ['aux_logits', 'transform_input']
    __constants__ = ['transform_input']

    #def __init__(self, num_classes=1000, aux_logits=False, transform_input=False,

    def __init__(self, num_classes=1000, transform_input=False,
                init_weights=True, blocks=None):

        super(GoogLeNet, self).__init__()
        if blocks is None:
            #blocks = [BasicConv2d, Inception, InceptionAux]
            blocks = [BasicConv2d, Inception]
        #assert len(blocks) == 3
        assert len(blocks) == 2

        conv_block = blocks[0]
        inception_block = blocks[1]
        #inception_aux_block = blocks[2]

        #self.aux_logits = aux_logits
        self.transform_input = transform_input

        self.conv1 = conv_block(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.conv2 = conv_block(64, 64, kernel_size=1)
        self.conv3 = conv_block(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        
        self.inception3a = inception_block(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = inception_block(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        
        self.inception4a = inception_block(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = inception_block(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = inception_block(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = inception_block(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = inception_block(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception5a = inception_block(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = inception_block(832, 384, 192, 384, 48, 128, 128)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(1024, 1000)
        
        self.module1 = nn.Sequential(
            self.conv1,
            self.maxpool1,
            self.conv2,
            self.conv3,
            self.maxpool2,
            self.inception3a,
            self.inception3b,
            self.maxpool3,
            self.inception4a
        )
        
        self.module2 = nn.Sequential(
            self.inception4b,
            self.inception4c,
            self.inception4d
        )

        self.module3 = nn.Sequential(
            self.inception4e,
            self.maxpool4,
            self.inception5a,
            self.inception5b,
            self.avgpool
        )
        
        self.module4 = nn.Sequential(
            self.dropout,
            self.fc
        )
        if init_weights:
            self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                init.kaiming_uniform_(m.weight)
            
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

    
    def forward(self, x):
        
        x = self.module1(x)
        x = self.module2(x)
        x = self.module3(x)
        x = torch.flatten(x, 1)
        x = self.module4(x)

        return x



class Inception(nn.Module):
    
    __constants__ = ['branch2', 'branch3', 'branch4']
    def __init__(self, in_channel, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj, 
                conv_block=None):
        
        super(Inception, self).__init__()
        
        if conv_block is None:
            conv_block = BasicConv2d

        self.branch1 = conv_block(in_channel, ch1x1, kernel_size=1)
    
        self.branch2 = nn.Sequential(
            conv_block(in_channel, ch3x3red, kernel_size=1),
            conv_block(ch3x3red, ch3x3, kernel_size=3, padding=1)
        )

        self.branch3 = nn.Sequential(
            conv_block(in_channel, ch5x5red, kernel_size=1),
            conv_block(ch5x5red, ch5x5, kernel_size=5, stride=1, padding=2)
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True),
            conv_block(in_channel, pool_proj, kernel_size=1)
        )

    def _forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        outputs = [branch1, branch2, branch3, branch4]

        return outputs
    
    def forward(self, x):
        outputs = self._forward(x)
        return torch.cat(outputs, 1)        # cat makes list of tensors to tensor
                                            # list[tensor, tensor] -> tensor


class BasicConv2d(nn.Module):


    def __init__(self, in_channel, out_channel, **kwargs):

        super(BasicConv2d, self).__init__()

        self.conv = nn.Conv2d(in_channel, out_channel, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channel, eps=0.001)


    def forward(x):

        x = self.conv(x)
        x = self.bn(x)

        return F.relu(x, inplace=True)


def googlenet(**kwargs):

    return GoogLeNet(**kwargs)
