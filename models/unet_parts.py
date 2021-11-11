"""U-net parts used for SuperPointNet_gauss2.py
"""
import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class double_conv(nn.Layer):
    """(conv => BN => ReLU) * 2"""

    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(nn.Conv2D(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2D(out_ch), paddle.nn.ReLU(
            ), nn.Conv2D(out_ch, out_ch, 3, padding=1), nn.BatchNorm2D(
            out_ch), paddle.nn.ReLU())

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Layer):

    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Layer):

    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(nn.MaxPool2D(2), double_conv(in_ch, out_ch)
            )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Layer):

    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear',
                align_corners=True)
        else:
            self.up = paddle.nn.layer.Conv2DTranspose(in_ch // 2, in_ch // 2,
                2, stride=2)
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.shape[2] - x1.shape[2]
        diffX = x2.shape[3] - x1.shape[3]
        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY -
            diffY // 2))
        x = paddle.concat([x2, x1], axis=1)
        x = self.conv(x)
        return x


class outconv(nn.Layer):

    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2D(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x
