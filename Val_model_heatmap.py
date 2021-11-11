"""This is the main validation interface using heatmap trick
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

from models.model_wrap import SuperPointFrontend_torch


@paddle.no_grad()
class Val_model_heatmap(SuperPointFrontend_torch):

    def __init__(self, config, device='gpu', verbose=False):
        self.config = config
        self.model = self.config['name']
        self.params = self.config['params']
        self.weights_path = self.config['pretrained']
        self.device = device
        self.nms_dist = self.config['nms']
        self.conf_thresh = self.config['detection_threshold']
        self.nn_thresh = self.config['nn_thresh']
        self.cell = 8
        self.cell_size = 8
        self.border_remove = 4
        self.sparsemap = None
        self.heatmap = None
        self.pts = None
        self.pts_subpixel = None
        self.pts_nms_batch = None
        self.desc_sparse_batch = None
        self.patches = None
        pass

    def loadModel(self):
        from utils.loader import modelLoader

        self.net = modelLoader(model=self.model, **self.params)

        checkpoint = paddle.load(self.weights_path)
        self.net.set_state_dict(checkpoint['model_state_dict'])

        logging.info('successfully load pretrained model from: %s', self.weights_path)
        pass

    def extract_patches(self, label_idx, img):

        from utils.losses import extract_patches
        patch_size = self.config['params']['patch_size']
        patches = extract_patches(label_idx.to(self.device),
                                  img.to(self.device),
                                  patch_size=patch_size)
        return patches
        pass

    def run(self, images):

        from Train_model_heatmap import Train_model_heatmap
        from utils.var_dim import toNumpy
        train_agent = Train_model_heatmap

        with paddle.no_grad():
            outs = self.net(images)
        semi = outs['semi']
        self.outs = outs

        channel = semi.shape[1]
        if channel == 64:
            heatmap = train_agent.flatten_64to1(semi, cell_size=self.cell_size)
        elif channel == 65:
            heatmap = flattenDetection(semi, tensor=True)

        heatmap_np = toNumpy(heatmap)
        self.heatmap = heatmap_np
        return self.heatmap
        pass

    def heatmap_to_pts(self):
        heatmap_np = self.heatmap

        pts_nms_batch = [self.getPtsFromHeatmap(h) for h in heatmap_np]
        self.pts_nms_batch = pts_nms_batch
        return pts_nms_batch

    def desc_to_sparseDesc(self):
        desc_sparse_batch = [self.sample_desc_from_points(self.outs['desc'], pts)
                             for pts in self.pts_nms_batch]
        self.desc_sparse_batch = desc_sparse_batch
        return desc_sparse_batch


if __name__ == '__main__':
    filename = 'configs/magicpoint_repeatability_heatmap.yaml'
    import yaml

    device = paddle.devive.set_device('gpu')

    paddle.set_default_dtype('float32')

    with open(filename, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    task = config['data']['dataset']

    from utils.loader import dataLoader_test as dataLoader

    data = dataLoader(config, dataset='hpatches')
    test_set, test_loader = data['test_set'], data['test_loader']

    val_agent = Val_model_heatmap(config['model'], device=device)

    for i, sample in tqdm(enumerate(test_loader)):
        if i > 1:
            break

        val_agent.loadModel()
        img = sample['image']
        print('image: ', img.shape)

        heatmap_batch = val_agent.run(img)
        pts = val_agent.heatmap_to_pts()
        print('pts[0]: ', pts[0].shape)
        print('pts: ', pts[0][:, :3])

        pts_subpixel = val_agent.soft_argmax_points(pts)
        print('subpixels: ', pts_subpixel[0][:, :3])

        desc_sparse = val_agent.desc_to_sparseDesc()
        print('desc_sparse[0]: ', desc_sparse[0].shape)
