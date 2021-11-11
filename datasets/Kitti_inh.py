"""dataloader for kitti raw dataset
inh --> (inherited from Coco)

Author: You-Yi Jau, Rui Zhu
Date: 2019/12/12
"""
import numpy as np
import paddle
from pathlib import Path
import paddle.io as data
import random
from settings import DATA_PATH
from settings import EXPER_PATH
from utils.tools import dict_update
import cv2
import logging
from numpy.linalg import inv
from utils.utils import homography_scaling_torch as homography_scaling
from utils.utils import filter_points
from datasets.Coco import Coco


class Kitti_inh(Coco):
    default_config = {'labels': None, 'cache_in_memory': False,
        'validation_size': 100, 'truncate': None, 'preprocessing': {
        'resize': [240, 320]}, 'num_parallel_calls': 10, 'augmentation': {
        'photometric': {'enable': False, 'primitives': 'all', 'params': {},
        'random_order': True}, 'homographic': {'enable': False, 'params': {
        }, 'valid_border_margin': 0}}, 'warped_pair': {'enable': False,
        'params': {}, 'valid_border_margin': 0}, 'homography_adaptation': {
        'enable': False}}

    def __init__(self, export=False, transform=None, task='train', seed=0,
        sequence_length=1, **config):
        self.config = self.default_config
        self.config = dict_update(self.config, config)
        self.transforms = transform
        self.action = 'train' if task == 'train' else 'val'
        self.root = Path(self.config['root'])
        root_split_txt = self.config.get('root_split_txt', None)
        self.root_split_txt = Path(self.root if root_split_txt is None else
            root_split_txt)
        scene_list_path = (self.root_split_txt / 'train.txt' if task ==\
            'train' else self.root_split_txt / 'val.txt')
        self.scenes = [(Path(self.root / folder[:-1]), Path(self.root /
            folder[:-4] / 'image_02' / 'data')) for folder in open(
            scene_list_path)]
        if self.config['labels']:
            self.labels = True
            self.labels_path = Path(self.config['labels'], task)
            print('load labels from: ', self.config['labels'] + '/' + task)
        else:
            self.labels = False
        self.crawl_folders(sequence_length)
        self.init_var()

    def crawl_folders(self, sequence_length):
        sequence_set = []
        demi_length = sequence_length - 1
        for scene, scene_img_folder in self.scenes:
            intrinsics = np.eye(3)
            image_paths = list(scene_img_folder.iterdir())
            names = [p.stem for p in image_paths]
            imgs = [str(p) for p in image_paths]
            if len(imgs) < sequence_length:
                continue
            for i in range(0, len(imgs) - demi_length):
                sample = None
                if self.labels:
                    p = Path(self.labels_path, scene.name, '{}.npz'.format(
                        names[i]))
                    if p.exists():
                        sample = {'intrinsics': intrinsics, 'image': [imgs[
                            i]], 'scene_name': scene.name, 'frame_ids': [i]}
                        sample.update({'name': [names[i]], 'points': [str(p)]})
                else:
                    sample = {'intrinsics': intrinsics, 'imgs': [imgs[i]],
                        'scene_name': scene.name, 'frame_ids': [i], 'name':
                        [names[i]]}
                if sample is not None:
                    for j in range(1, demi_length + 1):
                        sample['image'].append(imgs[i + j])
                        sample['frame_ids'].append(i + j)
                    sequence_set.append(sample)
        random.shuffle(sequence_set)
        self.samples = sequence_set
        logging.info('Finished crawl_folders for KITTI.')

    def get_img_from_sample(self, sample):
        imgs_path = sample['imgs']
        return str(imgs_path[0])

    def get_from_sample(self, entry, sample):
        return str(sample[entry][0])

    def format_sample(self, sample):
        sample_fix = {}
        if self.labels:
            entries = ['image', 'points', 'name']
            for entry in entries:
                sample_fix[entry] = self.get_from_sample(entry, sample)
        else:
            sample_fix['image'] = str(sample['imgs'][0])
            sample_fix['name'] = str(sample['scene_name'] + '/' + sample[
                'name'][0])
            sample_fix['scene_name'] = str(sample['scene_name'])
        return sample_fix


if __name__ == '__main__':
    pass
