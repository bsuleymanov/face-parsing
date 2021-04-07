import torch
from torch import nn
import torch.nn.functional as F


def conv_block(in_channels, out_channels, maxpool=None):
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    ]
    if maxpool is not None:
        layers.append(nn.MaxPool2d(maxpool))
    return nn.Sequential(*layers)


class Deconv2d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Deconv2d, self).__init__()
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv = conv_block(in_channels, out_channels)

    def forward(self, fmap1, fmap2):
        fmap2 = self.up(fmap2)
        #diff_y = fmap2.size(2) - fmap1.size(2)
        #diff_x = fmap2.size(3) - fmap1.size(3)
        #fmap1 = F.pad(fmap1, [diff_x // 2, diff_x - diff_x // 2,
        #                      diff_y // 2, diff_y - diff_y // 2])
        offset = fmap2.size(2) - fmap1.size(2)
        padding = 2 * [offset // 2, offset // 2]
        fmap1 = F.pad(fmap1, padding)
        fmap_cat = torch.cat([fmap1, fmap2], 1)
        #print(fmap1.size(), fmap2.size())
        return self.conv(fmap_cat)


class UNet(nn.Module):
    def __init__(self, feature_scale=2, n_classes=19, in_channels=3):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.feature_scale = feature_scale

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale / 2) for x in filters]

        self.conv1 = conv_block(self.in_channels, filters[0], 2)
        self.conv2 = conv_block(filters[0], filters[1], 2)
        self.conv3 = conv_block(filters[1], filters[2], 2)
        self.conv4 = conv_block(filters[2], filters[3], 2)

        self.center = conv_block(filters[3], filters[4] // self.feature_scale)

        self.deconv1 = Deconv2d(filters[4], filters[3] // self.feature_scale)
        self.deconv2 = Deconv2d(filters[3], filters[2] // self.feature_scale)
        self.deconv3 = Deconv2d(filters[2], filters[1] // self.feature_scale)
        self.deconv4 = Deconv2d(filters[1], filters[0])

        self.last_layer = nn.Conv2d(filters[0], n_classes, 1)

    def forward(self, image):
        down1 = self.conv1(image)
        down2 = self.conv2(down1)
        down3 = self.conv3(down2)
        down4 = self.conv4(down3)
        center = self.center(down4)
        out = self.deconv1(down4, center)
        #print(out.size(), down3.size())
        out = self.deconv2(down3, out)
        out = self.deconv3(down2, out)
        out = self.deconv4(down1, out)
        out = self.last_layer(out)
        return out
