import torch
import torch.nn as nn
import numpy as np
import sys
import os
sys.path.append(os.getcwd())
from backbone.cnn_block import Choice_Shuffle_Block, Choice_Shuffle_Block_x, Identity

__all__ = ['se_spos_shufflenet']
channel = [16,
           64, 64, 64, 64,
           160, 160, 160, 160,
           320, 320, 320, 320, 320, 320, 320, 320,
           640, 640, 640, 640]
last_channel = 1024


class SinglePath_OneShot(nn.Module):
    def __init__(self, dataset, num_classes, input_channels=3, resize=False, layers=20):
        super(SinglePath_OneShot, self).__init__()
        if dataset == 'cifar10' and not resize:
            first_stride = 1
            self.downsample_layers = [4, 8]
        elif dataset == 'imagenet' or resize:
            first_stride = 2
            self.downsample_layers = [0, 4, 8, 16]
        self.num_classes = num_classes
        self.layers = layers
        self.kernel_list = ['id', 3, 5, 7] # map_list: {'0':'id', '1':3, '2':5, '3':7}

        # stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, channel[0], kernel_size=3, stride=first_stride, padding=1, bias=False),
            nn.BatchNorm2d(channel[0], affine=False),
            nn.ReLU6(inplace=True)
        )
        # choice_block
        self.choice_block = nn.ModuleList([])
        for i in range(layers):
            if i in self.downsample_layers:
                stride = 2
                inp, oup = channel[i], channel[i + 1]
            else:
                stride = 1
                inp, oup = channel[i] // 2, channel[i + 1]
            layer_cb = nn.ModuleList([])
            for j in self.kernel_list:
                if j == 'x':
                    layer_cb.append(Choice_Shuffle_Block_x(inp, oup, stride=stride))
                elif j == 'id':
                    layer_cb.append(Identity(inp, oup, stride=stride, flag=False))
                else:
                    layer_cb.append(Choice_Shuffle_Block(inp, oup, kernel=j, stride=stride))
            self.choice_block.append(layer_cb)
        # last_conv
        self.last_conv = nn.Sequential(
            nn.Conv2d(channel[-1], last_channel, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(last_channel, affine=False),
            nn.ReLU6(inplace=True)
        )
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        # self.global_pooling = nn.AvgPool2d(7)
        self.classifier = nn.Linear(last_channel, self.num_classes, bias=False)
        self._initialize_weights()

    def forward(self, x, choice=np.random.randint(4, size=20)): # random sampling from choice list
        x = x. float()
        x = self.stem(x)
        
        # repeat
        for i, j in enumerate(choice):
            # print(x.shape)
            x = self.choice_block[i][j](x)

        x = self.last_conv(x)
        x = self.global_pooling(x)
        x = x.view(-1, last_channel)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if 'first' in name:
                    nn.init.normal_(m.weight, 0, 0.01)
                else:
                    nn.init.normal_(m.weight, 0, 1.0 / m.weight.shape[1])
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class SinglePath_Network(nn.Module):
    def __init__(self, dataset, resize, num_classes, layers, choice):
        super(SinglePath_Network, self).__init__()
        if dataset == 'cifar10' and not resize:
            first_stride = 1
            self.downsample_layers = [4, 8]
        elif dataset == 'imagenet' or resize:
            first_stride = 2
            self.downsample_layers = [0, 4, 8, 16]
        self.num_classes = num_classes
        self.layers = layers
        self.kernel_list = ['id', 3, 5, 7]

        # stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, channel[0], kernel_size=3, stride=first_stride, padding=1, bias=False),
            nn.BatchNorm2d(channel[0]),
            nn.ReLU6(inplace=True)
        )
        # choice_block
        self.choice_block = nn.ModuleList([])
        for i in range(layers):
            if i in self.downsample_layers:
                stride = 2
                inp, oup = channel[i], channel[i + 1]
            else:
                stride = 1
                inp, oup = channel[i] // 2, channel[i + 1]
            if choice[i] == 0:
                self.choice_block.append(Identity(inp, oup, stride=stride, flag=False))
            else:
                self.choice_block.append(Choice_Shuffle_Block(inp, oup, kernel=self.kernel_list[choice[i]], stride=stride, supernet=False))
        # last_conv
        self.last_conv = nn.Sequential(
            nn.Conv2d(channel[-1], last_channel, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(last_channel),
            nn.ReLU6(inplace=True)
        )
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        # self.global_pooling = nn.AvgPool2d(7)
        self.classifier = nn.Linear(last_channel, self.num_classes, bias=False)
        self._initialize_weights()

    def forward(self, x):
        x = self.stem(x)
        # repeat
        for i in range(self.layers):
            x = self.choice_block[i](x)
        x = self.last_conv(x)
        x = self.global_pooling(x)
        x = x.view(-1, last_channel)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if 'first' in name:
                    nn.init.normal_(m.weight, 0, 0.01)
                else:
                    nn.init.normal_(m.weight, 0, 1.0 / m.weight.shape[1])
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

def se_spos_shufflenet(dataset='imagenet', num_classes=2, input_channels=3, resize=False, layers=20):
    return SinglePath_OneShot(dataset=dataset, num_classes=num_classes, input_channels=3, resize=resize, layers=layers)

if __name__ == '__main__':
    net = SinglePath_OneShot(dataset='imagenet', num_classes=6)
    print(net)
    inputs = torch.randn(4,3,224,224)
    choice = list(np.random.randint(4, size=20))
    outputs = net(inputs, choice)
    print(outputs.shape)
    # from thop import profile
    # macs, params = profile(net, inputs=(inputs,choice))
    # print("Flops: {}, Params: {}".format(macs / 1e9, params / 1e6))