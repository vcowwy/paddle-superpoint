"""
Adapted from https://github.com/rpautrat/SuperPoint/blob/master/superpoint/datasets/synthetic_dataset.py
"""
import numpy as np
import tarfile
import random
import logging
import multiprocessing
import cv2
import shutil
from tqdm import tqdm
from imageio import imread
from pathlib import Path

import paddle.io as data
import paddle

from utils.tools import dict_update
from datasets import synthetic_dataset
from settings import DEBUG as debug
from settings import DATA_PATH
from settings import SYN_TMPDIR

TMPDIR = SYN_TMPDIR


def load_as_float(path):
    return imread(path).astype(np.float32) / 255


class SyntheticDataset_gaussian(data.Dataset):

    default_config = {'primitives': 'all', 'truncate': {},
        'validation_size': -1, 'test_size': -1, 'on-the-fly': False,
        'cache_in_memory': False, 'suffix': None,
        'add_augmentation_to_test_set': False, 'num_parallel_calls': 10,
        'generation': {'split_sizes': {'training': 10000, 'validation': 200,
        'test': 500}, 'image_size': [960, 1280], 'random_seed': 0, 'params':
        {'generate_background': {'min_kernel_size': 150, 'max_kernel_size':
        500, 'min_rad_ratio': 0.02, 'max_rad_ratio': 0.031}, 'draw_stripes':
        {'transform_params': (0.1, 0.1)}, 'draw_multiple_polygons': {
        'kernel_boundaries': (50, 100)}}}, 'preprocessing': {'resize': [240,
        320], 'blur_size': 11}, 'augmentation': {'photometric': {'enable': 
        False, 'primitives': 'all', 'params': {}, 'random_order': True},
        'homographic': {'enable': False, 'params': {},
        'valid_border_margin': 0}}}
    if debug == True:
        drawing_primitives = ['draw_checkerboard']
    else:
        drawing_primitives = ['draw_lines', 'draw_polygon',
            'draw_multiple_polygons', 'draw_ellipses', 'draw_star',
            'draw_checkerboard', 'draw_stripes', 'draw_cube', 'gaussian_noise']
    print(drawing_primitives)

    def parse_primitives(self, names, all_primitives):
        p = all_primitives if names == 'all' else names if isinstance(names,
            list) else [names]
        assert set(p) <= set(all_primitives)
        return p

    def __init__(self, seed=None, task='train', sequence_length=3,
        transform=None, target_transform=None, getPts=False, warp_input=\
        False, **config):
        from utils.homographies import sample_homography_np as sample_homography
        from utils.photometric import ImgAugTransform
        from utils.photometric import customizedTransform
        from utils.utils import compute_valid_mask
        from utils.utils import inv_warp_image
        from utils.utils import warp_points
        paddle.set_default_dtype('float32')
        np.random.seed(seed)
        random.seed(seed)
        self.config = self.default_config
        self.config = dict_update(self.config, dict(config))
        self.transform = transform
        self.sample_homography = sample_homography
        self.compute_valid_mask = compute_valid_mask
        self.inv_warp_image = inv_warp_image
        self.warp_points = warp_points
        self.ImgAugTransform = ImgAugTransform
        self.customizedTransform = customizedTransform
        self.enable_photo_train = self.config['augmentation']['photometric'][
            'enable']
        self.enable_homo_train = self.config['augmentation']['homographic'][
            'enable']
        self.enable_homo_val = False
        self.enable_photo_val = False
        self.action = 'training' if task == 'train' else 'validation'
        self.cell_size = 8
        self.getPts = getPts
        self.gaussian_label = False
        if self.config['gaussian_label']['enable']:
            self.gaussian_label = True
        self.pool = multiprocessing.Pool(6)
        primitives = self.parse_primitives(config['primitives'], self.
            drawing_primitives)
        basepath = Path(DATA_PATH, 'synthetic_shapes' + ('_{}'.format(
            config['suffix']) if config['suffix'] is not None else ''))
        basepath.mkdir(parents=True, exist_ok=True)
        splits = {s: {'images': [], 'points': []} for s in [self.action]}
        for primitive in primitives:
            tar_path = Path(basepath, '{}.tar.gz'.format(primitive))
            if not tar_path.exists():
                self.dump_primitive_data(primitive, tar_path, self.config)
            logging.info('Extracting archive for primitive {}.'.format(
                primitive))
            logging.info(f'tar_path: {tar_path}')
            tar = tarfile.open(tar_path)
            temp_dir = Path(TMPDIR)
            tar.extractall(path=temp_dir)
            tar.close()
            truncate = self.config['truncate'].get(primitive, 1)
            path = Path(temp_dir, primitive)
            for s in splits:
                e = [str(p) for p in Path(path, 'images', s).iterdir()]
                f = [p.replace('images', 'points') for p in e]
                f = [p.replace('.png', '.npy') for p in f]
                splits[s]['images'].extend(e[:int(truncate * len(e))])
                splits[s]['points'].extend(f[:int(truncate * len(f))])
        for s in splits:
            perm = np.random.RandomState(0).permutation(len(splits[s][
                'images']))
            for obj in ['images', 'points']:
                splits[s][obj] = np.array(splits[s][obj])[perm].tolist()
        self.crawl_folders(splits)

    def crawl_folders(self, splits):
        sequence_set = []
        for img, pnts in zip(splits[self.action]['images'], splits[self.
            action]['points']):
            sample = {'image': img, 'points': pnts}
            sequence_set.append(sample)
        self.samples = sequence_set

    def putGaussianMaps(self, center, accumulate_confid_map):
        crop_size_y = self.params_transform['crop_size_y']
        crop_size_x = self.params_transform['crop_size_x']
        stride = self.params_transform['stride']
        sigma = self.params_transform['sigma']
        grid_y = crop_size_y / stride
        grid_x = crop_size_x / stride
        start = stride / 2.0 - 0.5
        xx, yy = np.meshgrid(range(int(grid_x)), range(int(grid_y)))
        xx = xx * stride + start
        yy = yy * stride + start
        d2 = (xx - center[0]) ** 2 + (yy - center[1]) ** 2
        exponent = d2 / 2.0 / sigma / sigma
        mask = exponent <= sigma
        cofid_map = np.exp(-exponent)
        cofid_map = np.multiply(mask, cofid_map)
        accumulate_confid_map += cofid_map
        accumulate_confid_map[accumulate_confid_map > 1.0] = 1.0
        return accumulate_confid_map

    def __getitem__(self, index):

        def checkSat(img, name=''):
            if img.max() > 1:
                print(name, img.max())
            elif img.min() < 0:
                print(name, img.min())

        def imgPhotometric(img):
            augmentation = self.ImgAugTransform(**self.config['augmentation'])
            img = img[:, :, np.newaxis]
            img = augmentation(img)
            cusAug = self.customizedTransform()
            img = cusAug(img, **self.config['augmentation'])
            return img

        def get_labels(pnts, H, W):
            labels = paddle.zeros([H, W]).requires_grad_(False)
            pnts_int = paddle.min(paddle.to_tensor(pnts.round(), dtype=paddle.int64), paddle.to_tensor([[W - 1, H - 1]], dtype=paddle.int64))
            labels[pnts_int[:, 1], pnts_int[:, 0]] = 1
            return labels

        def get_label_res(H, W, pnts):
            quan = lambda x: paddle.to_tensor(x.round(), dtype=paddle.int64)
            labels_res = paddle.zeros([H, W, 2]).requires_grad_(False)
            labels_res[quan(pnts)[:, (1)], quan(pnts)[:, (0)], :
                ] = pnts - pnts.round()
            labels_res = labels_res.transpose(1, 2).transpose(0, 1)
            return labels_res
        from datasets.data_tools import np_to_tensor
        from utils.utils import filter_points
        from utils.var_dim import squeezeToNumpy
        sample = self.samples[index]
        img = load_as_float(sample['image'])
        H, W = img.shape[0], img.shape[1]
        self.H = H
        self.W = W
        pnts = np.load(sample['points'])
        pnts = paddle.to_tensor(pnts, dtype=paddle.float32)
        pnts = paddle.stack((pnts[:, (1)], pnts[:, (0)]), axis=1)
        pnts = filter_points(pnts, paddle.to_tensor([W, H]))
        sample = {}
        labels_2D = get_labels(pnts, H, W)
        sample.update({'labels_2D': labels_2D.unsqueeze(0)})
        if self.config['augmentation']['photometric']['enable_train'
            ] and self.action == 'training' or self.config['augmentation'][
            'photometric']['enable_val'] and self.action == 'validation':
            img = imgPhotometric(img)
        else:
            pass
        if not (self.config['augmentation']['homographic']['enable_train'] and
            self.action == 'training' or self.config['augmentation'][
            'homographic']['enable_val'] and self.action == 'validation'):
            from numpy.linalg import inv
            if self.transform is not None:
                img = self.transform(img)
            sample['image'] = img
            valid_mask = self.compute_valid_mask(
                paddle.to_tensor([H, W]), inv_homography=paddle.eye(3)
            )
            sample.update({'valid_mask': valid_mask})
            labels_res = get_label_res(H, W, pnts)
            pnts_post = pnts
        else:
            from utils.utils import homography_scaling_torch as homography_scaling
            from numpy.linalg import inv
            homography = self.sample_homography(np.array([2, 2]), shift=-1,
                **self.config['augmentation']['homographic']['params'])
            homography = inv(homography)
            homography = paddle.to_tensor(homography, dtype=paddle.float32)
            inv_homography = homography.inverse()
            img = paddle.to_tensor(img)
            warped_img = self.inv_warp_image(img.squeeze(), inv_homography,
                mode='bilinear')
            warped_img = warped_img.squeeze().numpy()
            warped_img = warped_img[:, :, np.newaxis]
            warped_pnts = self.warp_points(pnts, homography_scaling(
                homography, H, W))
            warped_pnts = filter_points(warped_pnts, paddle.to_tensor([W, H]))
            if self.transform is not None:
                warped_img = self.transform(warped_img)
            sample['image'] = warped_img
            valid_mask = self.compute_valid_mask(paddle.to_tensor([H, W]),
                inv_homography=inv_homography, erosion_radius=self.config[
                'augmentation']['homographic']['valid_border_margin'])
            sample.update({'valid_mask': valid_mask})
            labels_2D = get_labels(warped_pnts, H, W)
            sample.update({'labels_2D': labels_2D.unsqueeze(0)})
            labels_res = get_label_res(H, W, warped_pnts)
            pnts_post = warped_pnts
        if self.gaussian_label:
            from datasets.data_tools import get_labels_bi
            labels_2D_bi = get_labels_bi(pnts_post, H, W)
            labels_gaussian = self.gaussian_blur(squeezeToNumpy(labels_2D_bi))
            labels_gaussian = np_to_tensor(labels_gaussian, H, W)
            sample['labels_2D_gaussian'] = labels_gaussian
        sample.update({'labels_res': labels_res})
        if self.config['warped_pair']['enable']:
            from datasets.data_tools import warpLabels
            homography = self.sample_homography(np.array([2, 2]), shift=-1,
                **self.config['warped_pair']['params'])
            homography = np.linalg.inv(homography)
            inv_homography = np.linalg.inv(homography)
            homography = paddle.to_tensor(homography, dtype=paddle.float32)
            inv_homography = paddle.to_tensor(inv_homography, dtype=paddle.float32)
            warped_img = paddle.to_tensor(img.type, dtype=paddle.float32)
            warped_img = self.inv_warp_image(warped_img.squeeze(),
                inv_homography, mode='bilinear').unsqueeze(0)
            if (self.enable_photo_train == True and self.action == 'train' or
                self.enable_photo_val and self.action == 'val'):
                warped_img = imgPhotometric(warped_img.numpy().squeeze())
                warped_img = paddle.to_tensor(warped_img, dtype=paddle.float32)
                pass
            warped_img = paddle.reshape(warped_img, shape=[-1, H, W])
            warped_set = warpLabels(pnts, H, W, homography, bilinear=True)
            warped_labels = warped_set['labels']
            warped_res = warped_set['res']
            warped_res = warped_res.transpose(1, 2).transpose(0, 1)
            if self.gaussian_label:
                warped_labels_bi = warped_set['labels_bi']
                warped_labels_gaussian = self.gaussian_blur(squeezeToNumpy(
                    warped_labels_bi))
                warped_labels_gaussian = np_to_tensor(warped_labels_gaussian,
                    H, W)
                sample['warped_labels_gaussian'] = warped_labels_gaussian
                sample.update({'warped_labels_bi': warped_labels_bi})
            sample.update({'warped_img': warped_img, 'warped_labels':
                warped_labels, 'warped_res': warped_res})
            valid_mask = self.compute_valid_mask(paddle.to_tensor([H, W]),
                inv_homography=inv_homography, erosion_radius=self.config[
                'warped_pair']['valid_border_margin'])
            sample.update({'warped_valid_mask': valid_mask})
            sample.update({'homographies': homography, 'inv_homographies':
                inv_homography})
        if self.getPts:
            sample.update({'pts': pnts})
        return sample

    def __len__(self):
        return len(self.samples)

    def gaussian_blur(self, image):
        aug_par = {'photometric': {}}
        aug_par['photometric']['enable'] = True
        aug_par['photometric']['params'] = self.config['gaussian_label'][
            'params']
        augmentation = self.ImgAugTransform(**aug_par)
        image = image[:, :, np.newaxis]
        heatmaps = augmentation(image)
        return heatmaps.squeeze()
