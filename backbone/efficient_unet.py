import torch
import torch.nn as nn
from torchvision.models.efficientnet import _efficientnet_conf
import torchvision.models.efficientnet as EfficientArch
from torchvision.models.feature_extraction import create_feature_extractor
from .decoder_block import *

__all__ = [
    "efficient_b0_unet",
    "efficient_b1_unet",
    "efficient_b2_unet",
    "efficient_b3_unet",
    "efficient_b4_unet",
    "efficient_b5_unet",
    "efficient_b6_unet",
    "efficient_b7_unet",
    "efficient_v2_s_unet",
    "efficient_v2_m_unet",
    "efficient_v2_l_unet",
]


class EfficientUnet(nn.Module):
    # efficientnet_name: (width_mult, depth_mult)
    setting_dict = {
        "efficientnet_b0": (1.0, 1.0),
        "efficientnet_b1": (1.0, 1.1),
        "efficientnet_b2": (1.1, 1.2),
        "efficientnet_b3": (1.2, 1.4),
        "efficientnet_b4": (1.4, 1.8),
        "efficientnet_b5": (1.6, 2.2),
        "efficientnet_b6": (1.8, 2.6),
        "efficientnet_b7": (2.0, 3.1),
        "efficientnet_v2_s": (None, None),
        "efficientnet_v2_m": (None, None),
        "efficientnet_v2_l": (None, None),
    }

    def __init__(
        self,
        efficientnet_name: str,
        num_classes: int = 7,
        weights: str = None,
        freeze_encoder: bool = True,
        **kwargs,
    ):
        super().__init__()

        self.freeze_encoder = freeze_encoder

        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
        )

        width_mult, depth_mult = self.setting_dict[efficientnet_name]

        inverted_residual_setting, _ = _efficientnet_conf(
            efficientnet_name, width_mult=width_mult, depth_mult=depth_mult
        )

        try:
            net: EfficientArch.EfficientNet = EfficientArch.__dict__[efficientnet_name](
                weights=weights
            )
        except:
            state_dict = torch.load(weights, map_location="cpu")
            net: EfficientArch.EfficientNet = EfficientArch.__dict__[
                efficientnet_name
            ]()

            net_state_dict = net.state_dict()
            for k in state_dict:
                if k in net_state_dict:
                    if state_dict[k].shape != net_state_dict[k].shape:
                        state_dict[k] = net_state_dict[k]

            msg = net.load_state_dict(state_dict, strict=False)
            print("=> loaded from checkpoint '{}' with msg {}".format(weights, msg))

        if efficientnet_name == "efficientnet_v2_s":
            return_nodes = {
                "1": "0",
                "2": "1",
                "3": "2",
                "5": "3",
                "6": "4",
            }
            nodes_dim = {
                "0": inverted_residual_setting[-6].out_channels,
                "1": inverted_residual_setting[-5].out_channels,
                "2": inverted_residual_setting[-4].out_channels,
                "3": inverted_residual_setting[-2].out_channels,
                "4": inverted_residual_setting[-1].out_channels,
            }
        else:
            return_nodes = {
                "1": "0",
                "2": "1",
                "3": "2",
                "5": "3",
                "7": "4",
            }
            nodes_dim = {
                "0": inverted_residual_setting[-7].out_channels,
                "1": inverted_residual_setting[-6].out_channels,
                "2": inverted_residual_setting[-5].out_channels,
                "3": inverted_residual_setting[-3].out_channels,
                "4": inverted_residual_setting[-1].out_channels,
            }

        self.encoder = create_feature_extractor(net.features, return_nodes)

        if self.freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.decoder4 = DecoderBlock(
            x_in_channel=nodes_dim["4"],
            out_channel=512,
            last_in_channel=nodes_dim["3"],
        )

        self.decoder3 = DecoderBlock(
            x_in_channel=512,
            out_channel=256,
            last_in_channel=nodes_dim["2"],
        )

        self.decoder2 = DecoderBlock(
            x_in_channel=256,
            out_channel=128,
            last_in_channel=nodes_dim["1"],
        )

        self.decoder1 = DecoderBlock(
            x_in_channel=128,
            out_channel=64,
            last_in_channel=nodes_dim["0"],
        )

        self.decoder0 = DecoderWithoutConcat(x_in_channel=64, out_channel=32)
        self.pred = nn.Conv2d(32, num_classes, kernel_size=1)
        nn.init.kaiming_normal_(self.pred.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        x = (x / 255 - self.mean) / self.std
        feature_dict = self.encoder(x)
        x0 = feature_dict["0"]
        x1 = feature_dict["1"]
        x2 = feature_dict["2"]
        x3 = feature_dict["3"]
        x4 = feature_dict["4"]

        d4 = self.decoder4(x4, last=x3)
        d3 = self.decoder3(d4, last=x2)
        d2 = self.decoder2(d3, last=x1)
        d1 = self.decoder1(d2, last=x0)
        d0 = self.decoder0(d1)
        pred = self.pred(d0)
        return pred


def efficient_b0_unet(num_classes=7, freeze_encoder=False, **kwargs):
    return EfficientUnet(
        efficientnet_name="efficientnet_b0",
        num_classes=num_classes,
        freeze_encoder=freeze_encoder,
        **kwargs,
    )


def efficient_b1_unet(num_classes=7, freeze_encoder=False, **kwargs):
    return EfficientUnet(
        efficientnet_name="efficientnet_b1",
        num_classes=num_classes,
        freeze_encoder=freeze_encoder,
        **kwargs,
    )


def efficient_b2_unet(num_classes=7, freeze_encoder=False, **kwargs):
    return EfficientUnet(
        efficientnet_name="efficientnet_b2",
        num_classes=num_classes,
        freeze_encoder=freeze_encoder,
        **kwargs,
    )


def efficient_b3_unet(num_classes=7, freeze_encoder=False, **kwargs):
    return EfficientUnet(
        efficientnet_name="efficientnet_b3",
        num_classes=num_classes,
        freeze_encoder=freeze_encoder,
        **kwargs,
    )


def efficient_b4_unet(num_classes=7, freeze_encoder=False, **kwargs):
    return EfficientUnet(
        efficientnet_name="efficientnet_b4",
        num_classes=num_classes,
        freeze_encoder=freeze_encoder,
        **kwargs,
    )


def efficient_b5_unet(num_classes=7, freeze_encoder=False, **kwargs):
    return EfficientUnet(
        efficientnet_name="efficientnet_b5",
        num_classes=num_classes,
        freeze_encoder=freeze_encoder,
        **kwargs,
    )


def efficient_b6_unet(num_classes=7, freeze_encoder=False, **kwargs):
    return EfficientUnet(
        efficientnet_name="efficientnet_b6",
        num_classes=num_classes,
        freeze_encoder=freeze_encoder,
        **kwargs,
    )


def efficient_b7_unet(num_classes=7, freeze_encoder=False, **kwargs):
    return EfficientUnet(
        efficientnet_name="efficientnet_b7",
        num_classes=num_classes,
        freeze_encoder=freeze_encoder,
        **kwargs,
    )


def efficient_v2_s_unet(num_classes=7, freeze_encoder=False, **kwargs):
    return EfficientUnet(
        efficientnet_name="efficientnet_v2_s",
        num_classes=num_classes,
        freeze_encoder=freeze_encoder,
        **kwargs,
    )


def efficient_v2_m_unet(num_classes=7, freeze_encoder=False, **kwargs):
    return EfficientUnet(
        efficientnet_name="efficientnet_v2_m",
        num_classes=num_classes,
        freeze_encoder=freeze_encoder,
        **kwargs,
    )


def efficient_v2_l_unet(num_classes=7, freeze_encoder=False, **kwargs):
    return EfficientUnet(
        efficientnet_name="efficientnet_v2_l",
        num_classes=num_classes,
        freeze_encoder=freeze_encoder,
        **kwargs,
    )


if __name__ == "__main__":
    from thop import profile

    # model = EfficientArch.efficientnet_b0()
    model = efficient_v2_l_unet()
    x = torch.randn([1, 3, 224, 224])
    macs, params = profile(model, inputs=(x,))
    print(macs / 1e9, params / 1e6)
