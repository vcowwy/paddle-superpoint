"""
Network to load pretrained model from Magicleap.
"""
import paddle
import paddle.nn as nn


def conv(in_planes, out_planes, kernel_size=3):
    return nn.Sequential(nn.Conv2D(in_planes, out_planes, kernel_size=\
        kernel_size, padding=(kernel_size - 1) // 2, stride=2), paddle.nn.ReLU())


def upconv(in_planes, out_planes):
    return nn.Sequential(paddle.nn.layer.Conv2DTranspose(in_planes, out_planes,
        kernel_size=4, stride=2, padding=1), paddle.nn.ReLU())


class SuperPointNet(paddle.nn.Layer):

    def __init__(self):
        super(SuperPointNet, self).__init__()
        self.relu = paddle.nn.ReLU()
        self.pool = paddle.nn.MaxPool2D(kernel_size=2, stride=2)
        c1, c2, c3, c4, c5, d1 = 64, 64, 128, 128, 256, 256
        self.conv1a = paddle.nn.Conv2D(1, c1, kernel_size=3, stride=1,
            padding=1)
        self.conv1b = paddle.nn.Conv2D(c1, c1, kernel_size=3, stride=1,
            padding=1)
        self.conv2a = paddle.nn.Conv2D(c1, c2, kernel_size=3, stride=1,
            padding=1)
        self.conv2b = paddle.nn.Conv2D(c2, c2, kernel_size=3, stride=1,
            padding=1)
        self.conv3a = paddle.nn.Conv2D(c2, c3, kernel_size=3, stride=1,
            padding=1)
        self.conv3b = paddle.nn.Conv2D(c3, c3, kernel_size=3, stride=1,
            padding=1)
        self.conv4a = paddle.nn.Conv2D(c3, c4, kernel_size=3, stride=1,
            padding=1)
        self.conv4b = paddle.nn.Conv2D(c4, c4, kernel_size=3, stride=1,
            padding=1)
        self.convPa = paddle.nn.Conv2D(c4, c5, kernel_size=3, stride=1,
            padding=1)
        self.convPb = paddle.nn.Conv2D(c5, 65, kernel_size=1, stride=1,
            padding=0)
        self.convDa = paddle.nn.Conv2D(c4, c5, kernel_size=3, stride=1,
            padding=1)
        self.convDb = paddle.nn.Conv2D(c5, d1, kernel_size=1, stride=1,
            padding=0)

    def forward(self, x):
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


class PoseExpNet(nn.Layer):

    def __init__(self, nb_ref_imgs=2, output_exp=False):
        super(PoseExpNet, self).__init__()
        self.nb_ref_imgs = nb_ref_imgs
        self.output_exp = output_exp
        conv_planes = [16, 32, 64, 128, 256, 256, 256]
        self.conv1 = conv(3 * (1 + self.nb_ref_imgs), conv_planes[0],
            kernel_size=7)
        self.conv2 = conv(conv_planes[0], conv_planes[1], kernel_size=5)
        self.conv3 = conv(conv_planes[1], conv_planes[2])
        self.conv4 = conv(conv_planes[2], conv_planes[3])
        self.conv5 = conv(conv_planes[3], conv_planes[4])
        self.conv6 = conv(conv_planes[4], conv_planes[5])
        self.conv7 = conv(conv_planes[5], conv_planes[6])
        self.pose_pred = nn.Conv2D(conv_planes[6], 6 * self.nb_ref_imgs,
            kernel_size=1, padding=0)
        if self.output_exp:
            upconv_planes = [256, 128, 64, 32, 16]
            self.upconv5 = upconv(conv_planes[4], upconv_planes[0])
            self.upconv4 = upconv(upconv_planes[0], upconv_planes[1])
            self.upconv3 = upconv(upconv_planes[1], upconv_planes[2])
            self.upconv2 = upconv(upconv_planes[2], upconv_planes[3])
            self.upconv1 = upconv(upconv_planes[3], upconv_planes[4])
            self.predict_mask4 = nn.Conv2D(upconv_planes[1], self.
                nb_ref_imgs, kernel_size=3, padding=1)
            self.predict_mask3 = nn.Conv2D(upconv_planes[2], self.
                nb_ref_imgs, kernel_size=3, padding=1)
            self.predict_mask2 = nn.Conv2D(upconv_planes[3], self.
                nb_ref_imgs, kernel_size=3, padding=1)
            self.predict_mask1 = nn.Conv2D(upconv_planes[4], self.
                nb_ref_imgs, kernel_size=3, padding=1)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, paddle.nn.Conv2D) or isinstance(m,
                paddle.nn.layer.Conv2DTranspose):
                paddle.nn.initializer.XavierUniform(m.weight.data)
                if m.bias is not None:
                    zeros_init_(m.bias)

    def forward(self, target_image, ref_imgs):
        assert len(ref_imgs) == self.nb_ref_imgs
        input = [target_image]
        input.extend(ref_imgs)
        input = paddle.concat(input, 1)
        out_conv1 = self.conv1(input)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3(out_conv2)
        out_conv4 = self.conv4(out_conv3)
        out_conv5 = self.conv5(out_conv4)
        out_conv6 = self.conv6(out_conv5)
        out_conv7 = self.conv7(out_conv6)
        pose = self.pose_pred(out_conv7)
        pose = pose.mean(3).mean(2)
        pose = paddle.reshape(0.01 * pose, shape=[pose.size(0), self.nb_ref_imgs, 6])
        if self.output_exp:
            out_upconv5 = self.upconv5(out_conv5)[:, :, 0:out_conv4.size(2),
                0:out_conv4.size(3)]
            out_upconv4 = self.upconv4(out_upconv5)[:, :, 0:out_conv3.size(
                2), 0:out_conv3.size(3)]
            out_upconv3 = self.upconv3(out_upconv4)[:, :, 0:out_conv2.size(
                2), 0:out_conv2.size(3)]
            out_upconv2 = self.upconv2(out_upconv3)[:, :, 0:out_conv1.size(
                2), 0:out_conv1.size(3)]
            out_upconv1 = self.upconv1(out_upconv2)[:, :, 0:input.size(2), 
                0:input.size(3)]
            exp_mask4 = nn.functional.sigmoid(self.predict_mask4(out_upconv4))
            exp_mask3 = nn.functional.sigmoid(self.predict_mask3(out_upconv3))
            exp_mask2 = nn.functional.sigmoid(self.predict_mask2(out_upconv2))
            exp_mask1 = nn.functional.sigmoid(self.predict_mask1(out_upconv1))
        else:
            exp_mask4 = None
            exp_mask3 = None
            exp_mask2 = None
            exp_mask1 = None
        if self.training:
            return [exp_mask1, exp_mask2, exp_mask3, exp_mask4], pose
        else:
            return exp_mask1, pose
