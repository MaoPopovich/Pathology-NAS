import torch
import torch.nn as nn
import torchvision.models.resnet as ResNetArch
from torchvision.models.feature_extraction import create_feature_extractor
import backbone.clip_model as clip
from .decoder_block import *

__all__ = [
    "resnet18_unet",
    "resnet34_unet",
    "resnet50_unet",
    "resnet101_unet",
    "resnet152_unet",
    "resnext50_32x4d_unet",
    "resnext101_32x8d_unet",
    "resnext101_64x4d_unet",
    "wide_resnet50_2_unet",
    "wide_resnet101_2_unet",
    "clip_resnet50_unet",
    "clip_resnet101_unet",
    "clip_resnet50x4_unet",
]


class ResUnet(nn.Module):
    setting_dict = {
        "RN50": [64, 256, 512, 1024, 2048],
        "RN101": [64, 256, 512, 1024, 2048],
        "RN50x4": [64, 256, 512, 1024, 2048],
        "resnet18": [64, 64, 128, 256, 512],
        "resnet34": [64, 64, 128, 256, 512],
        "resnet50": [64, 256, 512, 1024, 2048],
        "resnet101": [64, 256, 512, 1024, 2048],
        "resnet152": [64, 256, 512, 1024, 2048],
        "resnext50_32x4d": [64, 256, 512, 1024, 2048],
        "resnext101_32x8d": [64, 256, 512, 1024, 2048],
        "resnext101_64x4d": [64, 256, 512, 1024, 2048],
        "wide_resnet50_2": [64, 256, 512, 1024, 2048],
        "wide_resnet101_2": [64, 256, 512, 1024, 2048],
    }

    def __init__(
        self,
        resnet_name: str,
        num_classes: int = 7,
        weights: str = None,
        from_clip: bool = False,
        freeze_encoder: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.freeze_encoder = freeze_encoder

        if from_clip:
            clip_model, _ = clip.load(resnet_name, device="cpu")
            self.encoder = create_feature_extractor(
                clip_model.visual,
                return_nodes={
                    "relu3": "0",
                    "layer1": "1",
                    "layer2": "2",
                    "layer3": "3",
                    "layer4": "4",
                },
            )
            self.register_buffer(
                "mean",
                torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1),
            )
            self.register_buffer(
                "std",
                torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1),
            )

        else:
            try:
                net: ResNetArch.ResNet = ResNetArch.__dict__[resnet_name](
                    weights=weights
                )
            except:
                ckpt = torch.load(weights, map_location="cpu")
                net: ResNetArch.ResNet = ResNetArch.__dict__[resnet_name]()
                msg = net.load_state_dict(ckpt, strict=False)
                print("=> loaded from checkpoint '{}' with msg {}".format(weights, msg))

            self.encoder = create_feature_extractor(
                net,
                return_nodes={
                    "relu": "0",
                    "layer1": "1",
                    "layer2": "2",
                    "layer3": "3",
                    "layer4": "4",
                },
            )

            self.register_buffer(
                "mean",
                torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            )
            self.register_buffer(
                "std",
                torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            )

        if self.freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        nodes_dim = {
            "0": self.setting_dict[resnet_name][0],
            "1": self.setting_dict[resnet_name][1],
            "2": self.setting_dict[resnet_name][2],
            "3": self.setting_dict[resnet_name][3],
            "4": self.setting_dict[resnet_name][4],
        }

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


def resnet18_unet(num_classes=7, freeze_encoder=False, **kwargs):
    return ResUnet(
        resnet_name="resnet18",
        num_classes=num_classes,
        freeze_encoder=freeze_encoder,
        **kwargs,
    )


def resnet34_unet(num_classes=7, freeze_encoder=False, **kwargs):
    return ResUnet(
        resnet_name="resnet34",
        num_classes=num_classes,
        freeze_encoder=freeze_encoder,
        **kwargs,
    )


def resnet50_unet(num_classes=7, freeze_encoder=False, **kwargs):
    return ResUnet(
        resnet_name="resnet50",
        num_classes=num_classes,
        freeze_encoder=freeze_encoder,
        **kwargs,
    )


def resnet101_unet(num_classes=7, freeze_encoder=False, **kwargs):
    return ResUnet(
        resnet_name="resnet101",
        num_classes=num_classes,
        freeze_encoder=freeze_encoder,
        **kwargs,
    )


def resnet152_unet(num_classes=7, freeze_encoder=False, **kwargs):
    return ResUnet(
        resnet_name="resnet152",
        num_classes=num_classes,
        freeze_encoder=freeze_encoder,
        **kwargs,
    )


def resnext50_32x4d_unet(num_classes=7, freeze_encoder=False, **kwargs):
    return ResUnet(
        resnet_name="resnext50_32x4d",
        num_classes=num_classes,
        freeze_encoder=freeze_encoder,
        **kwargs,
    )


def resnext101_32x8d_unet(num_classes=7, freeze_encoder=False, **kwargs):
    return ResUnet(
        resnet_name="resnext101_32x8d",
        num_classes=num_classes,
        freeze_encoder=freeze_encoder,
        **kwargs,
    )


def resnext101_64x4d_unet(num_classes=7, freeze_encoder=False, **kwargs):
    return ResUnet(
        resnet_name="resnext101_64x4d",
        num_classes=num_classes,
        freeze_encoder=freeze_encoder,
        **kwargs,
    )


def wide_resnet50_2_unet(num_classes=7, freeze_encoder=False, **kwargs):
    return ResUnet(
        resnet_name="wide_resnet50_2",
        num_classes=num_classes,
        freeze_encoder=freeze_encoder,
        **kwargs,
    )


def wide_resnet101_2_unet(num_classes=7, freeze_encoder=False, **kwargs):
    return ResUnet(
        resnet_name="wide_resnet101_2",
        num_classes=num_classes,
        freeze_encoder=freeze_encoder,
        **kwargs,
    )


def clip_resnet50_unet(num_classes=7, freeze_encoder=False, **kwargs):
    return ResUnet(
        resnet_name="RN50",
        num_classes=num_classes,
        freeze_encoder=freeze_encoder,
        from_clip=True,
        **kwargs,
    )


def clip_resnet101_unet(num_classes=7, freeze_encoder=False, **kwargs):
    return ResUnet(
        resnet_name="RN101",
        num_classes=num_classes,
        freeze_encoder=freeze_encoder,
        from_clip=True,
        **kwargs,
    )


def clip_resnet50x4_unet(num_classes=7, freeze_encoder=False, **kwargs):
    return ResUnet(
        resnet_name="RN50x4",
        num_classes=num_classes,
        freeze_encoder=freeze_encoder,
        from_clip=True,
        **kwargs,
    )


if __name__ == "__main__":
    from thop import profile

    model = resnet50_unet()
    # scripted_model = torch.jit.script(model)
    x = torch.randn([1, 3, 224, 224])
    macs, params = profile(model, inputs=(x,))
    print(macs / 1e9, params / 1e6)
    print(model(x).size())
