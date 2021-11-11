"""script for subpixel experiment (not tested)
"""
import numpy as np

import paddle
import logging
from tqdm import tqdm
from pathlib import Path

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
from Train_model_frontend import Train_model_frontend


class Train_model_subpixel(Train_model_frontend):
    default_config = {'train_iter': 170000,
                      'save_interval': 2000,
                      'tensorboard_interval': 200,
                      'model': {'subpixel': {'enable': False}}}

    def __init__(self, config, save_path=Path('.'), device='gpu', verbose=False):
        print('using: Train_model_subpixel')
        self.config = self.default_config
        self.config = dict_update(self.config, config)
        self.device = device
        self.save_path = save_path
        self.cell_size = 8
        self.max_iter = config['train_iter']
        self._train = True
        self._eval = True
        pass

    def print(self):
        print('hello')

    def loadModel(self):
        model = self.config['model']['name']
        params = self.config['model']['params']
        print('model: ', model)
        net = modelLoader(model=model, **params)

        logging.info('=> setting adam solver')

        optimizer = self.adamOptim(net, lr=self.config['model']['learning_rate'])

        n_iter = 0
        if self.config['retrain'] == True:
            logging.info('New model')
            pass
        else:
            path = self.config['pretrained']
            mode = '' if path[:-3] == '.pdiparams' else 'full'
            logging.info('load pretrained model from: %s', path)
            net, optimizer, n_iter = pretrainedLoader(net, optimizer, n_iter, path, mode=mode, full_path=True)
            logging.info('successfully load pretrained model from: %s', path)

        def setIter(n_iter):
            if self.config['reset_iter']:
                logging.info('reset iterations to 0')
                n_iter = 0
            return n_iter
        self.net = net
        self.optimizer = optimizer
        self.n_iter = setIter(n_iter)
        pass

    def train_val_sample(self, sample, n_iter=0, train=False):
        task = 'train' if train else 'val'
        tb_interval = self.config['tensorboard_interval']

        losses, tb_imgs, tb_hist = {}, {}, {}
        img, labels_2D, mask_2D = sample['image'], sample['labels_2D'], sample['valid_mask']

        labels_res = sample['labels_res']

        batch_size, H, W = img.shape[0], img.shape[2], img.shape[3]
        self.batch_size = batch_size

        Hc = H // self.cell_size
        Wc = W // self.cell_size

        self.optimizer.zero_grad()

        label_idx = labels_2D[...].nonzero()

        from utils.losses import extract_patches

        patch_size = self.config['model']['params']['patch_size']
        patches = extract_patches(label_idx
                                  img,
                                  patch_size=patch_size)

        patch_channels = self.config['model']['params'].get('subpixel_channel', 1)
        if patch_channels == 2:
            patch_heat = extract_patches(label_idx,
                                         img,
                                         patch_size=patch_size)

        def label_to_points(labels_res, points):
            labels_res = labels_res.transpose(1, 2).transpose(2, 3).unsqueeze(1)
            points_res = labels_res[points[:, (0)], points[:, (1)], points[:, (2)], points[:, (3)], :]
            return points_res

        points_res = label_to_points(labels_res, label_idx)

        num_patches_max = 500

        pred_res = self.net(patches[:num_patches_max, ...])

        def get_loss(points_res, pred_res):
            loss = points_res - pred_res
            loss = paddle.norm(loss, p=2, axis=-1).mean()
            return loss

        loss = get_loss(points_res[:num_patches_max, ...],
                        pred_res)
        self.loss = loss

        losses.update({'loss': loss})
        tb_hist.update({'points_res_0': points_res[:, 0]})
        tb_hist.update({'points_res_1': points_res[:, 1]})
        tb_hist.update({'pred_res_0': pred_res[:, 0]})
        tb_hist.update({'pred_res_1': pred_res[:, 1]})
        tb_imgs.update({'patches': patches[:, ...].unsqueeze(1)})
        tb_imgs.update({'img': img})

        losses.update({'loss': loss})
        if train:
            loss.backward()
            self.optimizer.step()

        self.tb_scalar_dict(losses, task)
        if n_iter % tb_interval == 0 or task == 'val':
            logging.info('current iteration: %d, tensorboard_interval: %d',
                         n_iter, tb_interval)
            self.tb_images_dict(task, tb_imgs, max_img=5)
            self.tb_hist_dict(task, tb_hist)

        return loss.item()

    def tb_images_dict(self, task, tb_imgs, max_img=5):
        for element in list(tb_imgs):
            for idx in range(tb_imgs[element].shape[0]):
                if idx >= max_img:
                    break
                self.writer.add_image(task + '-' + element + '/%d' % idx,
                    tb_imgs[element][idx, ...], self.n_iter)

    def tb_hist_dict(self, task, tb_dict):
        for element in list(tb_dict):
            self.writer.add_histogram(task + '-' + element, tb_dict[element], self.n_iter)
        pass


if __name__ == '__main__':
    filename = 'configs/magicpoint_shapes_subpix.yaml'
    import yaml

    device = paddle.device.set_device('gpu')

    paddle.set_default_dtype('float32')
    with open(filename, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    from utils.loader import dataLoader as dataLoader

    task = config['data']['dataset']

    data = dataLoader(config, dataset=task, warp_input=True)

    train_loader, val_loader = data['train_loader'], data['val_loader']

    train_agent = Train_model_subpixel(config, device=device)
    train_agent.print()

    from visualdl import LogWriter

    writer = LogWriter()
    train_agent.writer = writer

    train_agent.train_loader = train_loader
    train_agent.val_loader = val_loader

    train_agent.loadModel()
    train_agent.dataParallel()
    try:
        train_agent.train()
    except KeyboardInterrupt:
        print('press ctrl + c, save model!')
        train_agent.saveModel()
        pass
