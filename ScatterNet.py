import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class SqueezeExcitation(nn.Module):
    def __init__(self,channels,squeeze_channels=None):
        if squeeze_channels is None:
            squeeze_channels = channels//8
        super(SqueezeExcitation,self).__init__()

        self.channels = channels
        self.fc1 = nn.Conv2d(channels,squeeze_channels,kernel_size=1)
        self.fc2 = nn.Conv2d(squeeze_channels,channels,kernel_size=1)

    def forward(self,x):
        out = F.avg_pool2d(x,x.size()[2:])
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        out = F.sigmoid(out)
        return x*out

class DownBlock(nn.Module):
    def __init__(self,inchannels,channels,activation=nn.ReLU, batchnorm=False,squeeze = False,residual=True):
        super(DownBlock,self).__init__()
        self.residual=residual
        self.activation1 = activation()
        self.activation2 = activation()
        self.activation3 = activation()
        self.downconv = nn.Conv2d(inchannels,channels,kernel_size=2,stride=2,padding=1)

        self.conv1 = nn.Conv2d(channels,channels,kernel_size=3,padding=1)
        self.conv2 = nn.Conv2d(channels,channels,kernel_size=3,padding=1)
        if (batchnorm):
            self.bnorm1 = nn.BatchNorm3d(channels)
            self.bnorm2 = nn.BatchNorm3d(channels)
            self.bnorm3 = nn.BatchNorm3d(channels)
        if (squeeze):
            self.squeeze = SqueezeExcitation(channels)
        else:
            self.squeeze = None
        self.batchnorm=batchnorm

    def forward(self,x):

        down = self.downconv(x)
        if (self.batchnorm):
            down = self.bnorm1(down)
        down = self.activation1(down)
        # print("Down",down.size())
        out = self.conv1(down)
        if (self.batchnorm):
            out = self.bnorm2(out)
        out = self.activation2(out)
        out = self.conv2(out)
        if (self.batchnorm):
            out = self.bnorm3(out)
        if self.squeeze is not None:
            out = self.squeeze(out)
        if self.residual:
            out += down
        out = self.activation3(out)
        return out


class ResBlock(nn.Module):
    def __init__(self,inchannels,channels,activation=nn.ReLU,batchnorm = False, squeeze = False,residual=True):
        super(ResBlock,self).__init__()
        self.residual = residual
        self.activation1 = activation()
        self.activation2 = activation()
        self.activation3 = activation()
        self.conv0 = nn.Conv2d(inchannels,channels,kernel_size=3,padding=1)
        if (batchnorm):
            self.bnorm1 = nn.BatchNorm3d(channels)
            self.bnorm2 = nn.BatchNorm3d(channels)
            self.bnorm3 = nn.BatchNorm3d(channels)
        self.conv1 = nn.ConvTranspose2d(channels,channels,kernel_size=3,padding=1)
        self.conv2 = nn.ConvTranspose2d(channels,channels,kernel_size=3,padding=1)
        self.batchnorm = batchnorm
        if squeeze:
            self.squeeze = SqueezeExcitation(channels)
        else:
            self.squeeze = None

    def forward(self,x):
        up = self.conv0(x)
        if self.batchnorm:
            up = self.bnorm1(up)
        up = self.activation1(up)
        # print("Up",up.size())
        out = self.conv1(up)
        if self.batchnorm:
            out = self.bnorm2(out)
        out = self.activation2(out)
        out = self.conv2(out)
        if self.batchnorm:
            out = self.bnorm3(out)
        if self.squeeze is not None:
            out = self.squeeze(out)
        if self.residual:
            out += up

        out = self.activation3(out)
        return out
class UpBlock(nn.Module):
    def __init__(self,inchannels,channels,activation=nn.ReLU,batchnorm = False, squeeze = False,residual=True):
        super(UpBlock,self).__init__()
        self.residual=residual
        self.activation1 = activation()
        self.activation2 = activation()
        self.activation3 = activation()
        self.upconv = nn.Conv2d(inchannels,channels,kernel_size=3,padding=1)
        if (batchnorm):
            self.bnorm1 = nn.BatchNorm3d(channels)
            self.bnorm2 = nn.BatchNorm3d(channels)
            self.bnorm3 = nn.BatchNorm3d(channels)
        self.conv1 = nn.ConvTranspose2d(channels,channels,kernel_size=3,padding=1)
        self.conv2 = nn.ConvTranspose2d(channels,channels,kernel_size=3,padding=1)
        self.batchnorm = batchnorm
        if squeeze:
            self.squeeze = SqueezeExcitation(channels)
        else:
            self.squeeze = None

    def forward(self,x):
        up = F.upsample(x,scale_factor=2)
        up = self.upconv(up)
        if self.batchnorm:
            up = self.bnorm1(up)
        up = self.activation1(up)
        # print("Up",up.size())
        out = self.conv1(up)
        if self.batchnorm:
            out = self.bnorm2(out)
        out = self.activation2(out)
        out = self.conv2(out)
        if self.batchnorm:
            out = self.bnorm3(out)
        if self.squeeze is not None:
            out = self.squeeze(out)
        if self.residual:
          out += up

        out = self.activation3(out)
        return out


class ConvertNet(nn.Module):
    def __init__(self,init_channels,activation= nn.ReLU):
        super(ConvertNet,self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1,init_channels,kernel_size=3,padding=1),activation(),
                                   nn.Conv2d(init_channels,1,kernel_size=3,padding=1),activation())

    def forward(self,x):
        return self.conv1(x)


class ScatterNet(nn.Module):
    def __init__(self,init_channels, layer_channels,batchnorm = False, squeeze = False, skip_first = False,activation = nn.ReLU,exp = False,residual=True):
        """

        :type exp: bool
        """
        super(ScatterNet,self).__init__()

        self.activation = activation
        self.conv1 = ConvertNet(init_channels,activation=activation)

        self.conv2 = ResBlock(1,layer_channels[0],activation=activation,batchnorm=batchnorm,squeeze=squeeze,residual=residual)
        self.upblocks = nn.ModuleList()
        self.downblocks = nn.ModuleList()
        previous_channels = layer_channels[0]
        for channels in layer_channels[1:]:
            self.downblocks.append(DownBlock(previous_channels, channels, self.activation,batchnorm,squeeze,residual=residual))
            previous_channels = channels

        self.mixBlock = nn.ModuleList()
        for channels in reversed(layer_channels[:-1]):
            self.mixBlock.append(nn.Sequential(nn.Conv2d(channels*2,channels,kernel_size=1),self.activation()))
            self.upblocks.append(UpBlock(previous_channels,channels,self.activation,batchnorm,squeeze,residual=residual))
            previous_channels = channels


        self.dconvFinal = nn.ConvTranspose2d(layer_channels[0],1,kernel_size=1,padding=0)
        self.skip_first = skip_first
        self.exp = exp

    def forward(self,x):
        if self.skip_first:
            level1 = x
        else:
            level1 = self.conv1(x)
        if self.exp:
            level1 = torch.exp(-level1)*2**(16)

        previous = self.conv2(level1)
        layers = [previous]
        xsize = x.size()
        for block in self.downblocks:
            previous = block(previous)
            layers.append(previous)

        layers = list(reversed(layers[:-1]))
        for block,shortcut,mixer in zip(self.upblocks,layers,self.mixBlock):
            previous = block(previous)
            psize = previous.size()
            ssize = shortcut.size()
            if (psize != ssize):
                diff = np.array(ssize,dtype=int) - np.array(psize,dtype=int)
                # print(diff)
                previous = F.pad(previous,(0,int(diff[-1]),0,int(diff[-2])),mode="replicate")
            # print(previous.size(),shortcut.size())
            previous = torch.cat([previous,shortcut],dim=1)
            previous = mixer(previous)


        previous = self.dconvFinal(previous)
        if self.skip_first:
            return previous

        if self.exp:
            previous = torch.clamp(level1-previous,min=1e-6)
            # previous = -torch.log(previous)
            # previous = -torch.log(torch.clamp(previous,1e-6,1))
        else:
            previous = previous+level1
        return previous








