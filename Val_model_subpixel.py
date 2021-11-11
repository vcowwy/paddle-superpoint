"""script for subpixel experiment (not tested)
"""
import numpy as np
from tqdm import tqdm
import logging
from pathlib import Path

import paddle
import paddle.optimizer
import paddle.io

from utils.loader import dataLoader
from utils.loader import modelLoader
from utils.loader import pretrainedLoader
from utils.tools import dict_update
from utils.utils import labels2Dto3D
from utils.utils import flattenDetection
from utils.utils import labels2Dto3D_flattened
from utils.utils import pltImshow
from utils.utils import saveImg
from utils.utils import precisionRecall_torch
from utils.utils import save_checkpoint


@paddle.no_grad()
class Val_model_subpixel(object):

    def __init__(self, config, device='gpu', verbose=False):
        self.config = config
        self.model = self.config['name']
        self.params = self.config['params']
        self.weights_path = self.config['pretrained']
        self.device = device
        pass

    def loadModel(self):
        from utils.loader import modelLoader
        self.net = modelLoader(model=self.model, **self.params)

        checkpoint = paddle.load(self.weights_path)
        self.net.load_dict(checkpoint['model_state_dict'])

        self.net = self.net.to(self.device)
        logging.info('successfully load pretrained model from: %s',
                     self.weights_path)
        pass

    def extract_patches(self, label_idx, img):
        from utils.losses import extract_patches
        patch_size = self.config['params']['patch_size']
        patches = extract_patches(label_idx.to(self.device),
                                  img.to(self.device),
                                  patch_size=patch_size)
        return patches
        pass

    def run(self, patches):
        with paddle.no_grad():
            pred_res = self.net(patches)
        return pred_res
        pass


if __name__ == '__main__':
    filename = 'configs/magicpoint_repeatability.yaml'
    import yaml

    device = 'cuda' if paddle.is_compiled_with_cuda() else 'cpu'
    device = device.replace('cuda', 'gpu')
    device = paddle.set_device(device)

    paddle.set_default_dtype('float32')

    with open(filename, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    task = config['data']['dataset']

    from utils.loader import dataLoader_test as dataLoader

    data = dataLoader(config, dataset='hpatches')
    test_set, test_loader = data['test_set'], data['test_loader']
    for i, sample in tqdm(enumerate(test_loader)):
        if i > 1:
            break

        val_agent = Val_model_subpixel(config['subpixel'], device=device)
        val_agent.loadModel()

        img = sample['image']
        print('image: ', img.shape)
        points = paddle.to_tensor([[1, 2], [3, 4]])

        def points_to_4d(points):
            num_of_points = points.shape[0]
            cols = paddle.to_tensor(paddle.zeros([num_of_points, 1]).requires_grad_(False), dtype=paddle.float32)
            points = paddle.concat((cols, cols, paddle.to_tensor(points, dtype=paddle.float32)), axis=1)
            return points
        label_idx = points_to_4d(points)

        patches = val_agent.extract_patches(label_idx, img)
        points_res = val_agent.run(patches)
