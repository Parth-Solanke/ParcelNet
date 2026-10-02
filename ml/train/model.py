from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class DecoderBlock(nn.Module):
    """UNet decoder block with skip connection concatenation and double convolution."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        if skip is not None:
            # Handle possible 1-pixel rounding mismatch
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=True)
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class MultiTaskCadastraNet(nn.Module):
    """
    Multi-Task Segmentation Architecture for CadastraAI.
    Backbone: ResNet-34 encoder adapted for 4 input channels [R, G, B, nDSM].
    Decoders & Heads:
      - Buildings mask (1 channel logit)
      - Roads mask (1 channel logit)
      - Parcel boundaries mask (1 channel logit)
      - Boundary distance map (1 channel regression)
      - Land-use classification (num_classes logits)
    """

    def __init__(
        self,
        in_channels: int = 4,
        num_landuse_classes: int = 9,
        pretrained: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_landuse_classes = num_landuse_classes

        # ResNet-34 encoder
        weights = models.ResNet34_Weights.DEFAULT if pretrained else None
        base_resnet = models.resnet34(weights=weights)

        # Adapt first conv layer to 4 channels
        original_conv = base_resnet.conv1
        new_conv = nn.Conv2d(
            in_channels,
            original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False,
        )

        with torch.no_grad():
            # Copy RGB weights
            new_conv.weight[:, :3, :, :] = original_conv.weight[:, :3, :, :]
            if in_channels > 3:
                # Copy mean RGB weights into 4th nDSM channel
                mean_rgb_weights = original_conv.weight[:, :3, :, :].mean(dim=1, keepdim=True)
                new_conv.weight[:, 3:4, :, :] = mean_rgb_weights
                # If extra channels exist beyond 4, copy mean as well
                for extra_ch in range(4, in_channels):
                    new_conv.weight[:, extra_ch : extra_ch + 1, :, :] = mean_rgb_weights

        self.initial = nn.Sequential(
            new_conv,
            base_resnet.bn1,
            base_resnet.relu,
        )
        self.maxpool = base_resnet.maxpool  # downsample /4

        # Encoder stages
        self.layer1 = base_resnet.layer1  # 64 channels, /4
        self.layer2 = base_resnet.layer2  # 128 channels, /8
        self.layer3 = base_resnet.layer3  # 256 channels, /16
        self.layer4 = base_resnet.layer4  # 512 channels, /32

        # Decoder stages
        self.dec4 = DecoderBlock(in_channels=512, skip_channels=256, out_channels=256)  # -> /16
        self.dec3 = DecoderBlock(in_channels=256, skip_channels=128, out_channels=128)  # -> /8
        self.dec2 = DecoderBlock(in_channels=128, skip_channels=64, out_channels=64)   # -> /4
        self.dec1 = DecoderBlock(in_channels=64, skip_channels=64, out_channels=32)    # -> /2
        self.final_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )  # -> full resolution

        # Shared feature dimension: 32
        # Specialized multi-task heads
        self.head_buildings = nn.Conv2d(32, 1, kernel_size=1)
        self.head_roads = nn.Conv2d(32, 1, kernel_size=1)
        self.head_boundaries = nn.Conv2d(32, 1, kernel_size=1)
        self.head_distance_map = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.head_landuse = nn.Conv2d(32, num_landuse_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        input_size = x.shape[2:]

        # Encoder path
        x0 = self.initial(x)        # 64 ch, /2
        x_mp = self.maxpool(x0)     # 64 ch, /4
        x1 = self.layer1(x_mp)      # 64 ch, /4
        x2 = self.layer2(x1)        # 128 ch, /8
        x3 = self.layer3(x2)        # 256 ch, /16
        x4 = self.layer4(x3)        # 512 ch, /32

        # Decoder path with skip connections
        d4 = self.dec4(x4, x3)      # 256 ch, /16
        d3 = self.dec3(d4, x2)      # 128 ch, /8
        d2 = self.dec2(d3, x1)      # 64 ch, /4
        d1 = self.dec1(d2, x0)      # 32 ch, /2
        feats = self.final_up(d1)   # 32 ch, full res

        if feats.shape[2:] != input_size:
            feats = F.interpolate(feats, size=input_size, mode="bilinear", align_corners=True)

        return {
            "buildings": self.head_buildings(feats),
            "roads": self.head_roads(feats),
            "boundaries": self.head_boundaries(feats),
            "distance_map": self.head_distance_map(feats),
            "landuse": self.head_landuse(feats),
        }
