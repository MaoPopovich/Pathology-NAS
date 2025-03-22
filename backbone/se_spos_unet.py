import numpy as np
import torch
import torch.nn as nn
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backbone.cnn_block import VGGBlock, Choice_VGGBlock, Identity
from typing import List
__all__ = ["se_spos_unet"]


class SE_SPOS_UNet(nn.Module):
    """
        code from UNet in AI-pathology and Single-Shot-One-Path(SPOS)
    """

    def __init__(self, num_classes: int, layers: int, input_channels: int = 3, base_size: int = 32,):
        super(SE_SPOS_UNet, self).__init__()
        nb_filter = [
            base_size,
            base_size * 2,
            base_size * 4,
            base_size * 8,
            base_size * 16,
        ]
        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1),
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1),
        )

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.kernel_list = ['id', 3, 5, 7]
        assert layers % 2 == 0
        self.layers = layers
        self.sample_layers = self.layers // 2    # 9 = 1 + 2 * 4 
        self.stem = VGGBlock(input_channels, nb_filter[0], nb_filter[0])

        self.choice_block = nn.ModuleList([])
        # DownSampling 
        for i in range(1, self.sample_layers+1):
            layer_cb = self._create_layer(nb_filter[i-1], nb_filter[i])
            self.choice_block.append(layer_cb)

        # UpSampling
        for i in range(self.sample_layers):
            layer_cb = self._create_layer(nb_filter[self.sample_layers-i-1] + nb_filter[self.sample_layers-i], nb_filter[self.sample_layers-i-1])   
            self.choice_block.append(layer_cb)
        
        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        self._initialize_weights()

    def _create_layer(self, in_channels, out_channels):
        layer_cb = nn.ModuleList([])
        for j in self.kernel_list:
            if j == 'id':
                layer_cb.append(Identity(in_channels, out_channels, stride=1, flag=True))
            else:
                layer_cb.append(Choice_VGGBlock(in_channels, out_channels, out_channels, kernel=j))
        return layer_cb
    
    def forward(self, x, choice=np.random.randint(4, size=8)):
        x = (x / 255 - self.mean) / self.std
        x_down = []
        x = self.stem(x)

        for i, j in enumerate(choice[:self.sample_layers]):
            x_down.append(x)
            x = self.choice_block[i][j](self.pool(x))
        for i, j in enumerate(choice[self.sample_layers:]):
            x = self.choice_block[i+self.sample_layers][j](torch.cat([x_down[self.sample_layers-i-1], self.up(x)], 1))
        output = self.final(x)
        return output    

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

class Subnet_Unet(nn.Module):
    def __init__(self, num_classes: int, layers: int = 8, input_channels: int = 3, base_size: int = 32, choice: List[int] = None):
        super(Subnet_Unet, self).__init__()
        nb_filter = [
            base_size,
            base_size * 2,
            base_size * 4,
            base_size * 8,
            base_size * 16,
        ]
        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1),
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1),
        )

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.kernel_list = ['id', 3, 5, 7]
        assert layers % 2 == 0
        self.layers = layers
        self.sample_layers = self.layers // 2    # 9 = 1 + 2 * 4 
        self.stem = VGGBlock(input_channels, nb_filter[0], nb_filter[0])

        self.choice_block = nn.ModuleList([])
        # DownSampling 
        for i in range(1, self.sample_layers+1):
            if choice[i-1] == 0:
                layer_down = Identity(nb_filter[i-1], nb_filter[i], stride=1, flag=True)
            else:
                layer_down = Choice_VGGBlock(nb_filter[i-1], nb_filter[i], nb_filter[i], kernel=self.kernel_list[choice[i-1]])
            self.choice_block.append(layer_down)

        # UpSampling
        for i in range(self.sample_layers):
            if choice[self.sample_layers+i] == 0:
                layer_up = Identity(nb_filter[self.sample_layers-i-1] + nb_filter[self.sample_layers-i], nb_filter[self.sample_layers-i-1], stride=1, flag=True)
            else:
                layer_up = Choice_VGGBlock(nb_filter[self.sample_layers-i-1] + nb_filter[self.sample_layers-i], nb_filter[self.sample_layers-i-1], nb_filter[self.sample_layers-i-1], kernel=self.kernel_list[choice[self.sample_layers+i]])
            self.choice_block.append(layer_up)
        
        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        self._initialize_weights()
    
    def forward(self, x):
        x = (x / 255 - self.mean) / self.std
        x_down = []
        x = self.stem(x)
        
        for i in range(self.sample_layers):
            x_down.append(x)
            x = self.choice_block[i](self.pool(x))
        for i in range(self.sample_layers):
            x = self.choice_block[i+self.sample_layers](torch.cat([x_down[self.sample_layers-i-1], self.up(x)], 1))
        output = self.final(x)
        return output    

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
def se_spos_unet(num_classes=7, layers=8, input_channels=3, base_size=32, **kwargs):
    return SE_SPOS_UNet(
        num_classes=num_classes, 
        layers=layers, 
        input_channels=input_channels, 
        base_size=32, 
        **kwargs)

def subnet_unet(num_classes=7, layers=8, input_channels=3, base_size=32, choice=None, **kwargs):
    return Subnet_Unet(
        num_classes=num_classes, 
        layers=layers, 
        input_channels=input_channels, 
        base_size=32, 
        choice=choice,
        **kwargs)


if __name__ == '__main__':
    model = SE_SPOS_UNet(num_classes=6, layers=8)
    x = torch.randn(4,3,224,224)
    choice = np.random.randint(4, size=8) # the number of choice block is 8
    output = model(x,choice)
    print(model)
    print(output.shape)
