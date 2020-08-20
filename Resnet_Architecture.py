#!/usr/bin/env python
# coding: utf-8

# # Implementing Resnet architecture in Pytorch

# In[6]:


import torch
import torch.nn as nn

from collections import OrderedDict
from functools import partial


# # Modified from https://github.com/FrancescoSaverioZuppichini/ResNet/blob/master/ResNet.ipynb

# ### Dynamic conv padding

class conv2dAuto(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.padding = (self.kernel_size[0]//2, self.kernel_size[0]//2) # dynamic padding

class conv2dtransposeAuto(nn.ConvTranspose2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.padding = (self.kernel_size[0]//2, self.kernel_size[0]//2) # dynamic padding


conv = partial(conv2dAuto, bias=False)
convt = partial(conv2dtransposeAuto, bias=False)

#conv_ = conv(in_channels=3, out_channels=32, kernel_size=3)
#print(conv_)
#del conv_
#conv_ = conv(in_channels=3, out_channels=32, kernel_size=1)
#print(conv_)
#del conv_

#conv_ = convt(in_channels=3, out_channels=32, kernel_size=3)
#print(conv_)
#del conv_
#conv_ = convt(in_channels=3, out_channels=32, kernel_size=1)
#print(conv_)
#del conv_


# ### Base Residual Block which can be subclassed

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_dropout=0, dropout_prob=0):
        super().__init__()
        self.in_channels, self.out_channels, self.use_dropout, self.dropout_prob  = in_channels, out_channels, use_dropout, dropout_prob
        self.block = nn.Identity()
        self.shortcut = nn.Identity()
        if self.use_dropout == 1:
            self.dropout = nn.Dropout(self.dropout_prob, inplace=True)

    def forward(self, x):
        residual = x
        x = self.block(x)
        if self.should_apply_shortcut:
            residual = self.shortcut(residual)
        x += residual
        x = nn.ReLU()(x)
        if self.use_dropout == 1:
            x = self.dropout(x)
        return x

    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.out_channels



#ResidualBlock(3,32)


# ### ResNet Residual Block which subclasses ResidualBlock to define shortcut

# - ResNet Residual block will also have decimation and expansion size to match the size of the node where it gets summed

class ResNetResidualBlock(ResidualBlock):
    def __init__(self, in_channels, out_channels, expansion=1, downsampling=1, conv=conv, convt=convt,
                   use_convt=0, upsampling=1, *args, **kwargs):
        super().__init__(in_channels, out_channels, use_dropout=kwargs.get('use_dropout', 0), dropout_prob=kwargs.get('dropout_prob', 0))
        self.expansion, self.downsampling, self.conv, self.convt, self.upsampling = expansion, downsampling,                                                                                     conv, convt, upsampling
        self.use_convt = use_convt
        # override self.shortcut of base class to define Residual connection
        #self.shortcut = nn.Sequential(OrderedDict(
        #    {'conv': nn.Conv2d(in_channels, self.expanded_channels, kernel_size=1, stride=self.downsampling, bias=False),
        #     'bn': nn.BatchNorm2d(self.expanded_channels)
        #    })) if self.should_apply_shortcut else None
        if self.use_convt==0:
            self.shortcut = nn.Sequential(OrderedDict(
                    {'conv': self.conv(in_channels, self.expanded_channels, kernel_size=1, stride=self.downsampling,\
                                       bias=False),
                     'bn': nn.BatchNorm2d(self.expanded_channels)
                    })) if self.should_apply_shortcut else None
        else:
            if self.upsampling > 1:
                self.output_padding = 1
            else:
                self.output_padding = 0
            self.shortcut = nn.Sequential(OrderedDict(
                    {'conv': self.convt(in_channels, self.expanded_channels, kernel_size=1, stride=self.upsampling,\
                                        bias=False, output_padding=self.output_padding),
                     'bn': nn.BatchNorm2d(self.expanded_channels)
                    })) if self.should_apply_shortcut else None

    @property
    def expanded_channels(self):
        return self.expansion * self.out_channels

    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.out_channels



#ResNetResidualBlock(32,64)


# ### Basic ResNet Block

# - Basic block is a 2 layers of conv/bn/relu
# first define function to just get conv/bn layer
def conv_bn(in_channels, out_channels, conv, *args, **kwargs):
    return nn.Sequential(OrderedDict({'conv': conv(in_channels, out_channels, *args, **kwargs),
                                      'bn': nn.BatchNorm2d(out_channels)}))


#conv_bn(32, 64, nn.Conv2d, kernel_size=3)

class ResNetBasicBlock(ResNetResidualBlock):
    expansion = 1
    def __init__(self, in_channels, out_channels, activation=nn.ReLU, *args, **kwargs):
        super().__init__(in_channels, out_channels, *args, **kwargs)
        use_dropout = kwargs.get('use_dropout', 0)
        dropout_prob = kwargs.get('dropout_prob', 0)
        #if use_dropout==1:
        #    activation = nn.ModuleList(nn.ReLU(), nn.Dropout(dropout_prob, inplace=True))
        #else:
        #    activation = nn.ModuleList(nn.ReLU())

        if self.use_convt==0:
            if use_dropout == 0:
                self.block = nn.Sequential(
                    conv_bn(self.in_channels, self.out_channels, conv=self.conv, bias=False, \
                            kernel_size=3, stride=self.downsampling),
                    activation(),
                    conv_bn(self.out_channels, self.expanded_channels, conv=self.conv, kernel_size=3, bias=False))
            else:
                self.block = nn.Sequential(
                    conv_bn(self.in_channels, self.out_channels, conv=self.conv, bias=False, \
                            kernel_size=3, stride=self.downsampling),
                    activation(),
                    nn.Dropout(dropout_prob, inplace=True),
                    conv_bn(self.out_channels, self.expanded_channels, conv=self.conv, kernel_size=3, bias=False))
        else:
            if use_dropout == 0:
                self.block = nn.Sequential(
                    conv_bn(self.in_channels, self.out_channels, conv=self.convt, bias=False, \
                            kernel_size=3, stride=self.upsampling, output_padding=self.output_padding),
                    activation(),
                    conv_bn(self.out_channels, self.expanded_channels, conv=self.convt, kernel_size=3, bias=False))
            else:
                self.block = nn.Sequential(
                    conv_bn(self.in_channels, self.out_channels, conv=self.convt, bias=False, \
                            kernel_size=3, stride=self.upsampling, output_padding=self.output_padding),
                    activation(),
                    nn.Dropout(dropout_prob, inplace=True),
                    conv_bn(self.out_channels, self.expanded_channels, conv=self.convt, kernel_size=3, bias=False))


#ResNetBasicBlock(32, 64, use_dropout=1, dropout_prob=0.2)


# ### Resnet Layer

# - Stack multiple Resnet blocks in succession
# - 1st block in layer has downsampling, remaing block does not downsample
class ResNetLayer(nn.Module):
    def __init__(self, in_channels, out_channels, block=ResNetBasicBlock, n=1, use_convt=0, *args, **kwargs):
        super().__init__()
        # first block in the layer can have downsampling
        # We perform downsampling directly by convolutional layers that have a stride of 2
        if use_convt==0:
            downsampling = 2 if in_channels != out_channels else 1
            upsampling = 1
        else:
            upsampling = 2 if in_channels != out_channels else 1
            downsampling = 1

        self.blocks = nn.Sequential(
                      block(in_channels, out_channels, *args, **kwargs, downsampling=downsampling, \
                            upsampling=upsampling, use_convt=use_convt),
                      *[block(out_channels*block.expansion, out_channels, downsampling=1, upsampling=1, use_convt=use_convt, \
                              *args, **kwargs) for _ in range(n-1)]
                      )


    def forward(self, x):
        x = self.blocks(x)
        return x



#ResNetLayer(64, 128, block=ResNetBasicBlock, n=3, use_dropout=1, dropout_prob=0.2)


# ### Encoder

# - stacking multiple layers with increasing feature size

class ResNetEncoder(nn.Module):
    def __init__(self, in_channels, blocks_sizes=[64, 128, 256, 512], deepths=[2,2,2,2],
                activation=nn.ReLU, block=ResNetBasicBlock, *args, **kwargs):
        # block_size: op channels for every layer
        # depth: number of blocks to be repeated in every layer
        super().__init__()
        self.blocks_sizes = blocks_sizes
        use_dropout = kwargs.get('use_dropout', 0)
        dropout_prob = kwargs.get('dropout_prob', 0)

        # first layer is 7x7 conv, bn, relu, maxpool with stride2 and kernel_size=3
        if use_dropout==0:
            self.gate = nn.Sequential(
                        nn.Conv2d(in_channels, self.blocks_sizes[0], kernel_size=7, stride=2, padding=3, bias=False),
                        nn.BatchNorm2d(self.blocks_sizes[0]),
                        activation(),
                        nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        else:
            self.gate = nn.Sequential(
                        nn.Conv2d(in_channels, self.blocks_sizes[0], kernel_size=7, stride=2, padding=3, bias=False),
                        nn.BatchNorm2d(self.blocks_sizes[0]),
                        activation(),
                        nn.Dropout(use_dropout, inplace='True'),
                        nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.in_out_block_sizes = list(zip(self.blocks_sizes,self.blocks_sizes[1::]))
        self.blocks = nn.ModuleList([
            ResNetLayer(blocks_sizes[0], blocks_sizes[0], n=deepths[0], activation=activation,
                        block=block,  *args, **kwargs),
            *[ResNetLayer(in_channels * block.expansion,
                          out_channels, n=n, activation=activation,
                          block=block, *args, **kwargs)
              for (in_channels, out_channels), n in zip(self.in_out_block_sizes, deepths[1:])]
        ])

    def forward(self, x):
        x = self.gate(x)
        for b in self.blocks:
            x = b(x)
        return x



# ### Decoder


class ResnetDecoder(nn.Module):
    """
    This class represents the tail of ResNet. It performs a global pooling and maps the output to the
    correct class by using a fully connected layer.
    """
    def __init__(self, in_features, n_classes):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.decoder = nn.Linear(in_features, n_classes)

    def forward(self, x):
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.decoder(x)
        return x


# # Resnet

class ResNet(nn.Module):

    def __init__(self, in_channels, n_classes, *args, **kwargs):
        super().__init__()
        self.encoder = ResNetEncoder(in_channels, *args, **kwargs)
        self.decoder = ResnetDecoder(self.encoder.blocks[-1].blocks[-1].expanded_channels, n_classes)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# We can now defined the five models proposed by the Authors, resnet18,34,50,101,152

def resnet18(in_channels, n_classes):
    return ResNet(in_channels, n_classes, block=ResNetBasicBlock, deepths=[2, 2, 2, 2], use_dropout=1, dropout_prob=0.2)

def resnet34(in_channels, n_classes):
    return ResNet(in_channels, n_classes, block=ResNetBasicBlock, deepths=[3, 4, 6, 3])

def resnet50(in_channels, n_classes):
    return ResNet(in_channels, n_classes, block=ResNetBottleNeckBlock, deepths=[3, 4, 6, 3])

def resnet101(in_channels, n_classes):
    return ResNet(in_channels, n_classes, block=ResNetBottleNeckBlock, deepths=[3, 4, 23, 3])

def resnet152(in_channels, n_classes):
    return ResNet(in_channels, n_classes, block=ResNetBottleNeckBlock, deepths=[3, 8, 36, 3])


# In[61]:


from torchsummary import summary

#model = resnet18(3, 1000)
#summary(model.cuda(), (3, 224, 224))

#import torchvision.models as models
#model = models.resnet18
#summary(model().cuda(), (3, 224, 224))


# #### good resource on understanding conv transpose
#  - https://www.cs.toronto.edu/~lczhang/360/lec/w05/autoencoder.html
#  - https://github.com/vdumoulin/conv_arithmetic
#  - https://towardsdatascience.com/is-the-transposed-convolution-layer-and-convolution-layer-the-same-thing-8655b751c3a1

# ### rough draft of decoder using resnet architecture
#  - fc layer 256 -> 7x7x512
#  - reshape to (, 512, 7, 7)
#  - 1st Resnet Basic Block (input: (,512,7,7) , output: (,512,7,7))
#  - 2nd Resnet Basic Block (input: (,512,7,7), output: (,256,14,14))
#  - 3rd Resnet Basic Block (input: (,256,14,14), output: (,128,28,28))
#  - 4th Resnet Basic Block (input: (,128,28,28), output: (,64,56,56))
#  - 5th Resnet Basic Block (input: (,64,56,56), output: (,32,112,112))
#  - convt layer (input: (,32,112,112), output: (,3,224,224))
#  - sigmoid layer
#  - MSE loss

#ResNetResidualBlock(64, 32, use_convt=1, upsampling=2)

#ResNetBasicBlock(64, 32, use_convt=1, upsampling=2, use_dropout=1, dropout_prob=0.2)

#ResNetLayer(64, 128, block=ResNetBasicBlock, n=3, use_convt=1, use_dropout=1, dropout_prob=0.2)


class ResNetTransposeEncoder(nn.Module):
    def __init__(self, in_channels, blocks_sizes=[512, 256, 128, 64], deepths=[2,2,2,2],
                activation=nn.ReLU, out_channels=3, block=ResNetBasicBlock, use_convt=1, *args, **kwargs):
        # block_size: op channels for every layer
        # depth: number of blocks to be repeated in every layer
        super().__init__()
        self.blocks_sizes = blocks_sizes
        use_dropout = kwargs.get('use_dropout', 0)
        dropout_prob = kwargs.get('dropout_prob', 0)

        # first layer takes embedded dimension to 7*7*512
        if use_dropout==0:
            self.gate = nn.Sequential(
                        nn.Linear(in_features=in_channels, out_features=7*7*512, bias=False),
                        nn.BatchNorm1d(num_features = 7*7*512),
                        activation())
        else:
            self.gate = nn.Sequential(
                        nn.Linear(in_features=in_channels, out_features=7*7*512, bias=False),
                        nn.BatchNorm1d(num_features = 7*7*512),
                        activation(),
                        nn.Dropout(dropout_prob, inplace=True))

        #self.fc = nn.Linear(in_features=in_channels, out_features=7*7*512, bias=False)
        #self.bn = nn.BatchNorm1d(7*7*512)
        #self.nl = activation()

        self.in_out_block_sizes = list(zip(self.blocks_sizes,self.blocks_sizes[1::]))
        self.blocks = nn.ModuleList([
            ResNetLayer(blocks_sizes[0], blocks_sizes[0], n=deepths[0], activation=activation,
                        block=block, use_convt=use_convt, *args, **kwargs),
            *[ResNetLayer(in_channels * block.expansion,
                          out_channels, n=n, activation=activation,
                          block=block, use_convt=use_convt, *args, **kwargs)
              for (in_channels, out_channels), n in zip(self.in_out_block_sizes, deepths[1:])]
        ])
        self.final_convt = convt(self.blocks[-1].blocks[-1].expanded_channels, out_channels, kernel_size=3, stride=2,                                 output_padding=1)
        self.Tanh = nn.Tanh()

    def forward(self, x):
        x = self.gate(x)
        x = x.view(x.shape[0],512,7,7)
        for b in self.blocks:
            x = b(x)
        x = self.final_convt(x)
        x = self.Tanh(x)
        return x


#resnet_transpose_enc = ResNetTransposeEncoder(in_channels=256, blocks_sizes=[512, 256, 128, 64, 32], deepths=[2,2,2,2,2],                                               use_convt=1,use_dropout=1, dropout_prob=0.2)


#print(resnet_transpose_enc)

#summary(resnet_transpose_enc.cuda(), (256,))

#x = torch.randn(2,256)
#y = resnet_transpose_enc(x.cuda())
#y.shape
