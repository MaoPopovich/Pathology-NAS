import torch.nn as nn
import torch


class DecoderBlock(nn.Module):
    def __init__(
        self,
        x_in_channel: int,
        out_channel: int,
        last_in_channel: int = 0,
    ):
        super().__init__()

        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.act = nn.ReLU(True)

        self.conv1 = nn.Conv2d(
            x_in_channel + last_in_channel,
            out_channel,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.bn1 = nn.BatchNorm2d(out_channel)

        self.conv2 = nn.Conv2d(
            out_channel,
            out_channel,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.bn2 = nn.BatchNorm2d(out_channel)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor, last: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = torch.cat([x, last], dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act(x)
        return x


class DecoderWithoutConcat(nn.Module):
    def __init__(
        self,
        x_in_channel: int,
        out_channel: int,
        last_in_channel: int = 0,
    ):
        super().__init__()

        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.act = nn.ReLU(True)

        self.conv1 = nn.Conv2d(
            x_in_channel + last_in_channel,
            out_channel,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.bn1 = nn.BatchNorm2d(out_channel)

        self.conv2 = nn.Conv2d(
            out_channel,
            out_channel,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.bn2 = nn.BatchNorm2d(out_channel)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act(x)
        return x
