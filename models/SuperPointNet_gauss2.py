"""latest version of SuperpointNet. Use it!

"""
import numpy as np

import paddle

from models.unet_parts import *


class SuperPointNet_gauss2(paddle.nn.Layer):
    """ Pytorch definition of SuperPoint Network. """

    def __init__(self, subpixel_channel=1):
        super(SuperPointNet_gauss2, self).__init__()
        c1, c2, c3, c4, c5, d1 = 64, 64, 128, 128, 256, 256
        det_h = 65
        self.inc = inconv(1, c1)
        self.down1 = down(c1, c2)
        self.down2 = down(c2, c3)
        self.down3 = down(c3, c4)
        self.relu = paddle.nn.ReLU()
        self.convPa = paddle.nn.Conv2D(c4, c5, kernel_size=3, stride=1,
            padding=1)
        self.bnPa = nn.BatchNorm2D(c5)
        self.convPb = paddle.nn.Conv2D(c5, det_h, kernel_size=1, stride=1,
            padding=0)
        self.bnPb = nn.BatchNorm2D(det_h)
        self.convDa = paddle.nn.Conv2D(c4, c5, kernel_size=3, stride=1,
            padding=1)
        self.bnDa = nn.BatchNorm2D(c5)
        self.convDb = paddle.nn.Conv2D(c5, d1, kernel_size=1, stride=1,
            padding=0)
        self.bnDb = nn.BatchNorm2D(d1)
        self.output = None

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        cPa = self.relu(self.bnPa(self.convPa(x4)))
        semi = self.bnPb(self.convPb(cPa))
        cDa = self.relu(self.bnDa(self.convDa(x4)))
        desc = self.bnDb(self.convDb(cDa))
        dn = paddle.norm(desc, p=2, axis=1) # Compute the norm.
        desc = paddle.divide(desc, paddle.unsqueeze(dn, 1))
        output = {'semi': semi, 'desc': desc}
        self.output = output
        return output

    def process_output(self, sp_processer):
        from utils.utils import flattenDetection
        output = self.output
        semi = output['semi']
        desc = output['desc']
        heatmap = flattenDetection(semi)
        heatmap_nms_batch = sp_processer.heatmap_to_nms(heatmap, tensor=True)
        outs = sp_processer.pred_soft_argmax(heatmap_nms_batch, heatmap)
        residual = outs['pred']
        outs = sp_processer.batch_extract_features(desc, heatmap_nms_batch,
            residual)
        output.update(outs)
        self.output = output
        return output


def get_matches(deses_SP):
    from models.model_wrap import PointTracker
    tracker = PointTracker(max_length=2, nn_thresh=1.2)
    f = lambda x: x.cpu().detach().numpy()
    matching_mask = tracker.nn_match_two_way(f(deses_SP[0]).T, f(deses_SP[1
        ]).T, nn_thresh=1.2)
    return matching_mask


def main():

    device = paddle.device.set_device('gpu')
    model = SuperPointNet_gauss2()
    model = model
    from paddle import summary
    summary(model, input_size=(1, 240, 320))
    image = paddle.zeros((2, 1, 120, 160)).requires_grad_(False)
    outs = model(image)
    print('outs: ', list(outs))
    from utils.print_tool import print_dict_attr
    print_dict_attr(outs, 'shape')
    from models.model_utils import SuperPointNet_process
    params = {'out_num_points': 500, 'patch_size': 5, 'device': device,
        'nms_dist': 4, 'conf_thresh': 0.015}
    sp_processer = SuperPointNet_process(**params)
    outs = model.process_output(sp_processer)
    print('outs: ', list(outs))
    print_dict_attr(outs, 'shape')
    import time
    from tqdm import tqdm
    iter_max = 50
    start = time.time()
    print('Start timer!')
    for i in tqdm(range(iter_max)):
        outs = model(image)
    end = time.time()
    print('forward only: ', iter_max / (end - start), ' iter/s')
    start = time.time()
    print('Start timer!')
    xs_SP, deses_SP, reses_SP = [], [], []
    for i in tqdm(range(iter_max)):
        outs = model(image)
        outs = model.process_output(sp_processer)
        xs_SP.append(outs['pts_int'].squeeze())
        deses_SP.append(outs['pts_desc'].squeeze())
        reses_SP.append(outs['pts_offset'].squeeze())
    end = time.time()
    print('forward + process output: ', iter_max / (end - start), ' iter/s')
    start = time.time()
    print('Start timer!')
    for i in tqdm(range(len(xs_SP))):
        get_matches([deses_SP[i][0], deses_SP[i][1]])
    end = time.time()
    print('nn matches: ', iter_max / (end - start), ' iters/s')


if __name__ == '__main__':
    main()
