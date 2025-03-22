import torch
import torch.nn as nn


class VGGBlock(nn.Module):
    def __init__(self, in_channels: int, middle_channels: int, out_channels: int):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out

class Choice_VGGBlock(nn.Module):
    def __init__(self, 
        in_channels: int, 
        middle_channels: int, 
        out_channels: int, 
        kernel: int, 
        supernet: bool = True):
        if supernet:
            self.affine = False
        else:
            self.affine = True
        super(Choice_VGGBlock, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, kernel, padding='same')
        self.bn1 = nn.BatchNorm2d(middle_channels, affine=self.affine)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, kernel, padding='same')
        self.bn2 = nn.BatchNorm2d(out_channels, affine=self.affine)
        self.se = SELayer(out_channels, reduction=8)

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)  # SE Attention
        out = self.relu(out)

        return out

def channel_shuffle(x):
    """
        code from https://github.com/megvii-model/SinglePathOneShot/src/Search/blocks.py#L124    (batch_size, channels, height, width) => (2, batch_size, channels // 2, height, width)
    """
    batchsize, num_channels, height, width = x.data.size()
    assert (num_channels % 4 == 0)
    x = x.reshape(batchsize * num_channels // 2, 2, height * width)
    x = x.permute(1, 0, 2)
    x = x.reshape(2, -1, num_channels // 2, height, width)
    return x[0], x[1]


class Choice_Shuffle_Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride, supernet=True):
        super(Choice_Shuffle_Block, self).__init__()
        padding = kernel // 2
        if supernet:
            self.affine = False
        else:
            self.affine = True
        self.stride = stride
        self.in_channels = in_channels
        self.mid_channels = out_channels // 2
        self.out_channels = out_channels - in_channels

        self.cb_main = nn.Sequential(
            # pw
            nn.Conv2d(self.in_channels, self.mid_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.mid_channels, affine=self.affine),
            nn.ReLU(inplace=True),
            # dw  N_group = channels_in = channels_out  [depthwise convolution]
            nn.Conv2d(self.mid_channels, self.mid_channels, kernel_size=kernel, stride=stride, padding=padding,
                      bias=False, groups=self.mid_channels),
            nn.BatchNorm2d(self.mid_channels, affine=self.affine),
            # pw_linear
            nn.Conv2d(self.mid_channels, self.out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.out_channels, affine=self.affine),
            nn.ReLU(inplace=True)
        )
        if stride == 2:
            self.cb_proj = nn.Sequential(
                # dw
                nn.Conv2d(self.in_channels, self.in_channels, kernel_size=kernel, stride=2, padding=padding,
                          bias=False, groups=self.in_channels),
                nn.BatchNorm2d(self.in_channels, affine=self.affine),
                # pw
                nn.Conv2d(self.in_channels, self.in_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(self.in_channels, affine=self.affine),
                nn.ReLU(inplace=True)
            )
        self.se = SELayer(out_channels, reduction=8)

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = channel_shuffle(x)  # split channel numbers into two parts? use similar op in Identity?
            y = torch.cat((self.cb_main(x1), x2), 1)
        else:
            y = torch.cat((self.cb_main(x), self.cb_proj(x)), 1)
        y = self.se(y)   # SE Attention
        return y 


class Choice_Shuffle_Block_x(nn.Module):
    def __init__(self, in_channels, out_channels, stride, supernet=True):
        super(Choice_Shuffle_Block_x, self).__init__()
        if supernet:
            self.affine = False
        else:
            self.affine = True
        self.stride = stride
        self.in_channels = in_channels
        self.mid_channels = out_channels // 2
        self.out_channels = out_channels - in_channels

        self.cb_main = nn.Sequential(
            # dw
            nn.Conv2d(self.in_channels, self.in_channels, kernel_size=3, stride=stride,
                      padding=1, bias=False, groups=self.in_channels),
            nn.BatchNorm2d(self.in_channels, affine=self.affine),
            # pw
            nn.Conv2d(self.in_channels, self.mid_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.mid_channels, affine=self.affine),
            nn.ReLU(inplace=True),
            # dw
            nn.Conv2d(self.mid_channels, self.mid_channels, kernel_size=3, stride=1,
                      padding=1, bias=False, groups=self.mid_channels),
            nn.BatchNorm2d(self.mid_channels, affine=self.affine),
            # pw
            nn.Conv2d(self.mid_channels, self.mid_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.mid_channels, affine=self.affine),
            nn.ReLU(inplace=True),
            # dw
            nn.Conv2d(self.mid_channels, self.mid_channels, kernel_size=3, stride=1,
                      padding=1, bias=False, groups=self.mid_channels),
            nn.BatchNorm2d(self.mid_channels, affine=self.affine),
            # pw
            nn.Conv2d(self.mid_channels, self.out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.out_channels, affine=self.affine),
            nn.ReLU(inplace=True)
        )
        if stride == 2:
            self.cb_proj = nn.Sequential(
                # dw
                nn.Conv2d(self.in_channels, self.in_channels, kernel_size=3, stride=2,
                          padding=1, groups=self.in_channels, bias=False),
                nn.BatchNorm2d(self.in_channels, affine=self.affine),
                # pw
                nn.Conv2d(self.in_channels, self.in_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(self.in_channels, affine=self.affine),
                nn.ReLU(inplace=True)
            )
        self.se = SELayer(out_channels, reduction=8)

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = channel_shuffle(x)
            y = torch.cat((self.cb_main(x1), x2), 1)
        else:
            y = torch.cat((self.cb_main(x), self.cb_proj(x)), 1)
        y = self.se(y)  # SE Attention
        return y

class Identity(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, flag=False):
        super(Identity, self).__init__()
        if flag: # 1x1 conv for channel conversion:flag=True->Unet, flag=False->ShuffleNet
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            if stride != 1: # channel conversion for the first layer of each block
                self.downsample = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False),
                    nn.BatchNorm2d(out_channels),
                )
            else:
                self.downsample = None
    
    def forward(self, x):
        if self.downsample is not None:
            x = self.downsample(x)
        return x

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        bs, nc, _, _ = x.size()
        y = self.avg_pool(x).view(bs, nc)
        y = self.fc(y).view(bs, nc, 1, 1)
        return x * y.expand_as(x)