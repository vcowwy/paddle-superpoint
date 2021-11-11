import paddle
import paddle.nn as nn


class SuperPointNet(paddle.nn.Layer):
    """ Pytorch definition of SuperPoint Network. """

    def __init__(self):
        super(SuperPointNet, self).__init__()

        def predict_flow(in_planes):
            return nn.Conv2D(in_planes, 2, kernel_size=3, stride=1, padding=1, bias_attr=True)

        def convrelu(in_channels, out_channels, kernel, padding):
            return nn.Sequential(
                nn.Conv2D(in_channels, out_channels, kernel, padding=padding),
                paddle.nn.ReLU())

        self.relu = paddle.nn.ReLU()
        self.pool = paddle.nn.MaxPool2D(kernel_size=2, stride=2, return_mask=True)
        self.unpool = paddle.nn.MaxPool2D(kernel_size=2, stride=2)
        c1, c2, c3, c4, c5, d1 = 64, 64, 128, 128, 256, 256
        det_h = 65
        gn = 64
        useGn = False
        self.reBn = True
        if self.reBn:
            print('model structure: relu - bn - conv')
        else:
            print('model structure: bn - relu - conv')
        if useGn:
            print('apply group norm!')
        else:
            print('apply batch norm!')
        self.conv1a = paddle.nn.Conv2D(1, c1, kernel_size=3, stride=1,
            padding=1)
        self.bn1a = nn.GroupNorm(gn, c1) if useGn else nn.BatchNorm2D(c1)
        self.conv1b = paddle.nn.Conv2D(c1, c1, kernel_size=3, stride=1,
            padding=1)
        self.bn1b = nn.GroupNorm(gn, c1) if useGn else nn.BatchNorm2D(c1)
        self.conv2a = paddle.nn.Conv2D(c1, c2, kernel_size=3, stride=1,
            padding=1)
        self.bn2a = nn.GroupNorm(gn, c2) if useGn else nn.BatchNorm2D(c2)
        self.conv2b = paddle.nn.Conv2D(c2, c2, kernel_size=3, stride=1,
            padding=1)
        self.bn2b = nn.GroupNorm(gn, c2) if useGn else nn.BatchNorm2D(c2)
        self.conv3a = paddle.nn.Conv2D(c2, c3, kernel_size=3, stride=1,
            padding=1)
        self.bn3a = nn.GroupNorm(gn, c3) if useGn else nn.BatchNorm2D(c3)
        self.conv3b = paddle.nn.Conv2D(c3, c3, kernel_size=3, stride=1,
            padding=1)
        self.bn3b = nn.GroupNorm(gn, c3) if useGn else nn.BatchNorm2D(c3)
        self.conv4a = paddle.nn.Conv2D(c3, c4, kernel_size=3, stride=1,
            padding=1)
        self.bn4a = nn.GroupNorm(gn, c4) if useGn else nn.BatchNorm2D(c4)
        self.conv4b = paddle.nn.Conv2D(c4, c4, kernel_size=3, stride=1,
            padding=1)
        self.bn4b = nn.GroupNorm(gn, c4) if useGn else nn.BatchNorm2D(c4)
        self.convPa = paddle.nn.Conv2D(c4, c5, kernel_size=3, stride=1,
            padding=1)
        self.bnPa = nn.GroupNorm(gn, c5) if useGn else nn.BatchNorm2D(c5)
        self.convPb = paddle.nn.Conv2D(c5, det_h, kernel_size=1, stride=1,
            padding=0)
        self.bnPb = nn.GroupNorm(det_h, det_h) if useGn else nn.BatchNorm2D(65)
        self.convDa = paddle.nn.Conv2D(c4, c5, kernel_size=3, stride=1,
            padding=1)
        self.bnDa = nn.GroupNorm(gn, c5) if useGn else nn.BatchNorm2D(c5)
        self.convDb = paddle.nn.Conv2D(c5, d1, kernel_size=1, stride=1,
            padding=0)
        self.bnDb = nn.GroupNorm(gn, d1) if useGn else nn.BatchNorm2D(d1)

    def forward(self, x, subpixel=False):

        if self.reBn:
            x = self.relu(self.bn1a(self.conv1a(x)))
            conv1 = self.relu(self.bn1b(self.conv1b(x)))
            x, ind1 = self.pool(conv1)
            x = self.relu(self.bn2a(self.conv2a(x)))
            conv2 = self.relu(self.bn2b(self.conv2b(x)))
            x, ind2 = self.pool(conv2)
            x = self.relu(self.bn3a(self.conv3a(x)))
            conv3 = self.relu(self.bn3b(self.conv3b(x)))
            x, ind3 = self.pool(conv3)
            x = self.relu(self.bn4a(self.conv4a(x)))
            x = self.relu(self.bn4b(self.conv4b(x)))
            cPa = self.relu(self.bnPa(self.convPa(x)))
            semi = self.bnPb(self.convPb(cPa))
            cDa = self.relu(self.bnDa(self.convDa(x)))
            desc = self.bnDb(self.convDb(cDa))
        else:
            x = self.bn1a(self.relu(self.conv1a(x)))
            x = self.bn1b(self.relu(self.conv1b(x)))
            x = self.pool(x)
            x = self.bn2a(self.relu(self.conv2a(x)))
            x = self.bn2b(self.relu(self.conv2b(x)))
            x = self.pool(x)
            x = self.bn3a(self.relu(self.conv3a(x)))
            x = self.bn3b(self.relu(self.conv3b(x)))
            x = self.pool(x)
            x = self.bn4a(self.relu(self.conv4a(x)))
            x = self.bn4b(self.relu(self.conv4b(x)))
            cPa = self.bnPa(self.relu(self.convPa(x)))
            semi = self.bnPb(self.convPb(cPa))
            cDa = self.bnDa(self.relu(self.convDa(x)))
            desc = self.bnDb(self.convDb(cDa))
        dn = paddle.norm(desc, p=2, axis=1) # Compute the norm.
        desc = desc.div(paddle.unsqueeze(dn, 1))
        output = {'semi': semi, 'desc': desc}
        if subpixel:
            pass
        return output


def forward_original(self, x):

    x = self.relu(self.conv1a(x))
    x = self.relu(self.conv1b(x))
    x = self.pool(x)
    x = self.relu(self.conv2a(x))
    x = self.relu(self.conv2b(x))
    x = self.pool(x)
    x = self.relu(self.conv3a(x))
    x = self.relu(self.conv3b(x))
    x = self.pool(x)
    x = self.relu(self.conv4a(x))
    x = self.relu(self.conv4b(x))
    cPa = self.relu(self.convPa(x))
    semi = self.convPb(cPa)
    cDa = self.relu(self.convDa(x))
    desc = self.convDb(cDa)
    dn = paddle.norm(desc, p=2, axis=1) # Compute the norm.
    desc = paddle.divide(desc, paddle.unsqueeze(dn, 1))
    return semi, desc


def conv(in_planes, out_planes, kernel_size=3):
    return nn.Sequential(nn.Conv2D(in_planes, out_planes, kernel_size=\
        kernel_size, padding=(kernel_size - 1) // 2, stride=2), paddle.nn.ReLU())


def upconv(in_planes, out_planes):
    return nn.Sequential(paddle.nn.layer.Conv2DTranspose(in_planes, out_planes,
        kernel_size=4, stride=2, padding=1), paddle.nn.ReLU())


if __name__ == '__main__':

    device = paddle.set_device('gpu')
    model = SuperPointNet()
    model = model
    from paddle import summary
    summary(model, input_size=(1, 224, 224))
