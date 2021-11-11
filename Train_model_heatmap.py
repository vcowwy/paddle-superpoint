"""This is the main training interface using heatmap trick
"""
import numpy as np
import logging
from pathlib import Path

import paddle
import paddle.optimizer
import paddle.nn as nn
import paddle.io

from utils.tools import dict_update
from utils.utils import precisionRecall_torch
from Train_model_frontend import Train_model_frontend


def thd_img(img, thd=0.015):
    img[img < thd] = 0
    img[img >= thd] = 1
    return img


def toNumpy(tensor):
    return tensor.detach().cpu().numpy()


def img_overlap(img_r, img_g, img_gray):
    img = np.concatenate((img_gray, img_gray, img_gray), axis=0)
    img[0, :, :] += img_r[0, :, :]
    img[1, :, :] += img_g[0, :, :]
    img[img > 1] = 1
    img[img < 0] = 0
    return img


class Train_model_heatmap(Train_model_frontend):

    default_config = {'train_iter': 170000,
                      'save_interval': 2000,
                      'tensorboard_interval': 200,
                      'model': {'subpixel': {'enable': False}},
                      'data': {'gaussian_label': {'enable': False}}}

    def __init__(self, config, save_path=Path('.'), device='gpu', verbose=False):
        print('Load Train_model_heatmap!!')

        self.config = self.default_config
        self.config = dict_update(self.config, config)
        print('check config!!', self.config)

        self.device = device
        self.save_path = save_path
        self._train = True
        self._eval = True
        self.cell_size = 8
        self.subpixel = False

        self.max_iter = config['train_iter']

        self.gaussian = False
        if self.config['data']['gaussian_label']['enable']:
            self.gaussian = True

        if self.config['model']['dense_loss']['enable']:
            print('use dense_loss!')
            from utils.utils import descriptor_loss
            self.desc_params = self.config['model']['dense_loss']['params']
            self.descriptor_loss = descriptor_loss
            self.desc_loss_type = 'dense'
        elif self.config['model']['sparse_loss']['enable']:
            print('use sparse_loss!')
            self.desc_params = self.config['model']['sparse_loss']['params']

            from utils.loss_functions.sparse_loss import batch_descriptor_loss_sparse
            self.descriptor_loss = batch_descriptor_loss_sparse
            self.desc_loss_type = 'sparse'
        self.printImportantConfig()
        pass

    def detector_loss(self, input, target, mask=None, loss_type='softmax'):

        if loss_type == 'l2':
            loss_func = nn.MSELoss(reduction="mean")
            loss = loss_func(input, target)
        elif loss_type == 'softmax':
            loss_func_BCE = nn.BCELoss(reduction='none').cuda()
            loss = loss_func_BCE(nn.functional.softmax(input, axis=1), target)
            loss = (loss.sum(dim=1) * mask).sum()
            loss = loss / (mask.sum() + 1e-10)
        return loss

    def train_val_sample(self, sample, n_iter=0, train=False):

        to_floatTensor = lambda x: paddle.to_tensor(x)

        task = 'train' if train else 'val'
        tb_interval = self.config['tensorboard_interval']
        if_warp = self.config['data']['warped_pair']['enable']

        self.scalar_dict, self.images_dict, self.hist_dict = {}, {}, {}
        img, labels_2D, mask_2D = sample['image'], \
                                  sample['labels_2D'], \
                                  sample['valid_mask']

        batch_size, H, W = img.shape[0], img.shape[2], img.shape[3]
        self.batch_size = batch_size
        det_loss_type = self.config['model']['detector_loss']['loss_type']

        Hc = H // self.cell_size
        Wc = W // self.cell_size

        if if_warp:
            img_warp, labels_warp_2D, mask_warp_2D = sample['warped_img'], \
                                                     sample['warped_labels'], \
                                                     sample['warped_valid_mask']

        if if_warp:
            mat_H, mat_H_inv = sample['homographies'], sample[
                'inv_homographies']

        self.optimizer.zero_grad()

        if train:
            outs = self.net(img
            semi, coarse_desc = outs['semi'], outs['desc']
            if if_warp:
                outs_warp = self.net(img_warp)
                semi_warp, coarse_desc_warp = outs_warp['semi'], outs_warp['desc']
        else:
            with paddle.no_grad():
                outs = self.net(img)
                semi, coarse_desc = outs['semi'], outs['desc']
                if if_warp:
                    outs_warp = self.net(img_warp)
                    semi_warp, coarse_desc_warp = outs_warp['semi'], outs_warp['desc']
                pass

        from utils.utils import labels2Dto3D

        if self.gaussian:
            labels_2D = sample['labels_2D_gaussian']
            if if_warp:
                warped_labels = sample['warped_labels_gaussian']
        else:
            labels_2D = sample['labels_2D']
            if if_warp:
                warped_labels = sample['warped_labels']

        add_dustbin = False
        if det_loss_type == 'l2':
            add_dustbin = False
        elif det_loss_type == 'softmax':
            add_dustbin = True

        labels_3D = paddle.to_tensor(labels2Dto3D(labels_2D,
                                 cell_size=self.cell_size,
                                 add_dustbin=add_dustbin), dtype=paddle.float32)
        mask_3D_flattened = self.getMasks(mask_2D,
                                          self.cell_size,
                                          device=self.device)

        loss_det = self.detector_loss(input=outs['semi'],
                                      target=labels_3D,
                                      mask=mask_3D_flattened,
                                      loss_type=det_loss_type)
        if if_warp:
            labels_3D = paddle.to_tensor(labels2Dto3D(
                warped_labels,
                cell_size=self.cell_size,
                add_dustbin=add_dustbin), dtype=paddle.float32)
            mask_3D_flattened = self.getMasks(
                mask_warp_2D,
                self.cell_size,
                device=self.device)
            loss_det_warp = self.detector_loss(
                input=outs_warp['semi'],
                target=labels_3D,
                mask=mask_3D_flattened,
                loss_type=det_loss_type)
        else:
            loss_det_warp = paddle.to_tensor([0], dtype=paddle.float32)

        mask_desc = mask_3D_flattened.unsqueeze(1)

        lambda_loss = self.config['model']['lambda_loss']

        if lambda_loss > 0:
            assert if_warp == True, 'need a pair of images'
            loss_desc, mask, positive_dist, negative_dist = (self.descriptor_loss(coarse_desc,
                                                                                  coarse_desc_warp,
                                                                                  mat_H,
                                                                                  mask_valid=mask_desc,
                                                                                  device=self.device, **self.desc_params))
        else:
            ze = paddle.to_tensor([0])
            loss_desc, positive_dist, negative_dist = ze, ze, ze

        loss = loss_det + loss_det_warp
        if lambda_loss > 0:
            loss += lambda_loss * loss_desc

        add_res_loss = False
        if add_res_loss and n_iter % 10 == 0:
            print('add_res_loss!!!')
            heatmap_org = self.get_heatmap(semi, det_loss_type)
            heatmap_org_nms_batch = self.heatmap_to_nms(
                self.images_dict,
                heatmap_org,
                name='heatmap_org')
            if if_warp:
                heatmap_warp = self.get_heatmap(semi_warp, det_loss_type)
                heatmap_warp_nms_batch = self.heatmap_to_nms(
                    self.images_dict,
                    heatmap_warp,
                    name='heatmap_warp')

            outs_res = self.get_residual_loss(
                sample['labels_2D'] * to_floatTensor(heatmap_org_nms_batch).unsqueeze(1),
                heatmap_org,
                sample['labels_res'],
                name='original_pred')

            loss_res_ori = (outs_res['loss'] ** 2).mean()

            if if_warp:
                outs_res_warp = self.get_residual_loss(
                    sample['warped_labels'] * to_floatTensor(heatmap_warp_nms_batch).unsqueeze(1),
                    heatmap_warp,
                    sample['warped_res'],
                    name='warped_pred')
                loss_res_warp = (outs_res_warp['loss'] ** 2).mean()
            else:
                loss_res_warp = paddle.to_tensor([0])
            loss_res = loss_res_ori + loss_res_warp

            loss += loss_res
            self.scalar_dict.update({'loss_res_ori': loss_res_ori,
                                     'loss_res_warp': loss_res_warp})

        self.loss = loss

        self.scalar_dict.update({'loss': loss,
                                 'loss_det': loss_det,
                                 'loss_det_warp': loss_det_warp,
                                 'positive_dist': positive_dist,
                                 'negative_dist': negative_dist})

        self.input_to_imgDict(sample, self.images_dict)

        if train:
            loss.backward()
            self.optimizer.step()

        if n_iter % tb_interval == 0 or task == 'val':
            logging.info('current iteration: %d, tensorboard_interval: %d',
                         n_iter, tb_interval)

            heatmap_org = self.get_heatmap(semi, det_loss_type)
            heatmap_org_nms_batch = self.heatmap_to_nms(
                self.images_dict,
                heatmap_org,
                name='heatmap_org')
            if if_warp:
                heatmap_warp = self.get_heatmap(semi_warp, det_loss_type)
                heatmap_warp_nms_batch = self.heatmap_to_nms(
                    self.images_dict, heatmap_warp, name='heatmap_warp')

            def update_overlap(images_dict, labels_warp_2D,
                heatmap_nms_batch, img_warp, name):

                from utils.draw import img_overlap

                nms_overlap = [img_overlap(toNumpy(labels_warp_2D[i]),
                                           heatmap_nms_batch[i],
                                           toNumpy(img_warp[i]))
                               for i in range(heatmap_nms_batch.shape[0])]
                nms_overlap = np.stack(nms_overlap, axis=0)
                images_dict.update({(name + '_nms_overlap'): nms_overlap})

            from utils.var_dim import toNumpy
            update_overlap(self.images_dict,
                           labels_2D,
                           heatmap_org_nms_batch[np.newaxis, ...],
                           img,
                           'original')
            update_overlap(self.images_dict,
                           labels_2D,
                           toNumpy(heatmap_org),
                           img,
                           'original_heatmap')
            if if_warp:
                update_overlap(self.images_dict,
                               labels_warp_2D,
                               heatmap_warp_nms_batch[np.newaxis, ...],
                               img_warp,
                               'warped')
                update_overlap(self.images_dict,
                               labels_warp_2D,
                               toNumpy(heatmap_warp),
                               img_warp,
                               'warped_heatmap')

            from utils.losses import do_log

            if self.gaussian:
                self.get_residual_loss(sample['labels_2D'],
                                       sample['labels_2D_gaussian'],
                                       sample['labels_res'],
                                       name='original_gt')

                if if_warp:
                    self.get_residual_loss(sample['warped_labels'],
                                           sample['warped_labels_gaussian'],
                                           sample['warped_res'],
                                           name='warped_gt')
            pr_mean = self.batch_precision_recall(to_floatTensor(
                heatmap_org_nms_batch[:, np.newaxis, ...]),
                sample['labels_2D'])

            print('pr_mean')
            self.scalar_dict.update(pr_mean)

            self.printLosses(self.scalar_dict, task)
            self.tb_images_dict(task, self.images_dict, max_img=2)
            self.tb_hist_dict(task, self.hist_dict)

        self.tb_scalar_dict(self.scalar_dict, task)

        return loss.item()

    def heatmap_to_nms(self, images_dict, heatmap, name):

        from utils.var_dim import toNumpy
        heatmap_np = toNumpy(heatmap)

        heatmap_nms_batch = [self.heatmap_nms(h) for h in heatmap_np]
        heatmap_nms_batch = np.stack(heatmap_nms_batch, axis=0)

        images_dict.update({(name + '_nms_batch'): heatmap_nms_batch[:, np.newaxis, ...]})

        return heatmap_nms_batch

    def get_residual_loss(self, labels_2D, heatmap, labels_res, name=''):
        if abs(labels_2D).sum() == 0:
            return
        outs_res = self.pred_soft_argmax(labels_2D, heatmap, labels_res,
                                         patch_size=5, device=self.device)
        self.hist_dict[name + '_resi_loss_x'] = outs_res['loss'][:, 0]
        self.hist_dict[name + '_resi_loss_y'] = outs_res['loss'][:, 1]
        err = abs(outs_res['loss']).mean(dim=0)

        var = abs(outs_res['loss']).std(dim=0)
        self.scalar_dict[name + '_resi_loss_x'] = err[0]
        self.scalar_dict[name + '_resi_loss_y'] = err[1]
        self.scalar_dict[name + '_resi_var_x'] = var[0]
        self.scalar_dict[name + '_resi_var_y'] = var[1]
        self.images_dict[name + '_patches'] = outs_res['patches']
        return outs_res

    @staticmethod
    def batch_precision_recall(batch_pred, batch_labels):
        precision_recall_list = []
        for i in range(batch_labels.shape[0]):
            precision_recall = precisionRecall_torch(batch_pred[i],
                                                     batch_labels[i])
            precision_recall_list.append(precision_recall)
        precision = np.mean(
            [precision_recall['precision']
             for precision_recall in precision_recall_list])
        recall = np.mean(
            [precision_recall['recall']
             for precision_recall in precision_recall_list])
        return {'precision': precision, 'recall': recall}

    @staticmethod
    def pred_soft_argmax(labels_2D, heatmap, labels_res, patch_size=5,
        device='gpu'):

        from utils.losses import norm_patches

        outs = {}

        from utils.losses import extract_patches
        from utils.losses import soft_argmax_2d

        label_idx = paddle.to_tensor(labels_2D[...].nonzero(), dtype=paddle.int64)
        patches = extract_patches(label_idx,
                                  heatmap,
                                  patch_size=patch_size)

        patches = norm_patches(patches)

        from utils.losses import do_log

        patches_log = do_log(patches)

        dxdy = soft_argmax_2d(patches_log, normalized_coordinates=False)
        dxdy = dxdy.squeeze(1)
        dxdy = dxdy - patch_size // 2

        def ext_from_points(labels_res, points):

            labels_res = labels_res.transpose(1, 2).transpose(2, 3).unsqueeze(1)
            points_res = labels_res[points[:, (0)], points[:, (1)], points[:, (2)], points[:, (3)], :]
            return points_res

        points_res = ext_from_points(labels_res, label_idx)

        outs['pred'] = dxdy
        outs['points_res'] = points_res

        outs['loss'] = dxdy - points_res
        outs['patches'] = patches
        return outs

    @staticmethod
    def flatten_64to1(semi, cell_size=8):

        from utils.d2s import DepthToSpace

        depth2space = DepthToSpace(cell_size)
        heatmap = depth2space(semi)
        return heatmap

    @staticmethod
    def heatmap_nms(heatmap, nms_dist=4, conf_thresh=0.015):

        from utils.utils import getPtsFromHeatmap

        heatmap = heatmap.squeeze()

        pts_nms = getPtsFromHeatmap(heatmap, conf_thresh, nms_dist)
        semi_thd_nms_sample = np.zeros_like(heatmap)
        semi_thd_nms_sample[pts_nms[1, :].astype(np.int), pts_nms[0, :].astype(np.int)] = 1
        return semi_thd_nms_sample


if __name__ == '__main__':
    filename = 'configs/superpoint_coco_train_heatmap.yaml'
    import yaml

    device = paddle.device.set_device('gpu')

    paddle.set_default_dtype('float32')
    with open(filename, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    from utils.loader import dataLoader as dataLoader
    task = config['data']['dataset']

    data = dataLoader(config, dataset=task, warp_input=True)

    train_loader, val_loader = data['train_loader'], data['val_loader']

    train_agent = Train_model_heatmap(config, device=device)

    train_agent.train_loader = train_loader

    train_agent.loadModel()
    train_agent.dataParallel()
    train_agent.train()

    #try:
    #    model_fe.train()
    #except KeyboardInterrupt:
    #    logging.info('ctrl + c is pressed. save model')
