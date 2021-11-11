"""
This script exports detection/ description using pretrained model.
"""
import argparse
import time
import csv
import yaml
import os
import logging
from pathlib import Path
import numpy as np
from imageio import imread
from tqdm import tqdm

from visualdl import LogWriter

import paddle
import paddle.optimizer
import paddle.io

from utils.utils import getWriterPath
from utils.utils import inv_warp_image_batch
from models.model_wrap import SuperPointFrontend_torch, PointTracker
## parameters
from settings import EXPER_PATH


def combine_heatmap(heatmap, inv_homographies, mask_2D, device='gpu'):
    heatmap = heatmap * mask_2D

    heatmap = inv_warp_image_batch(heatmap, inv_homographies[0, :, :, :],
                                   device=device, mode='bilinear')

    mask_2D = inv_warp_image_batch(mask_2D, inv_homographies[0, :, :, :],
                                   device=device, mode='bilinear')
    heatmap = paddle.sum(heatmap, axis=0)
    mask_2D = paddle.sum(mask_2D, axis=0)
    return heatmap / mask_2D
    pass


def export_descriptor(config, output_dir, args):

    from utils.loader import get_save_path
    from utils.var_dim import squeezeToNumpy

    device = paddle.device.set_device('gpu')
    logging.info('train on device: %s', device)
    with open(os.path.join(output_dir, 'config.yml'), 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    writer = LogWriter(getWriterPath(task=args.command, date=True))
    save_path = get_save_path(output_dir)
    save_output = save_path / '../predictions'
    os.makedirs(save_output, exist_ok=True)

    outputMatches = True
    subpixel = config['model']['subpixel']['enable']
    patch_size = config['model']['subpixel']['patch_size']

    from utils.loader import dataLoader_test as dataLoader
    task = config['data']['dataset']
    data = dataLoader(config, dataset=task)
    test_set, test_loader = data['test_set'], data['test_loader']

    from utils.print_tool import datasize
    datasize(test_loader, config, tag='test')

    from utils.loader import get_module
    Val_model_heatmap = get_module('', config['front_end_model'])
    val_agent = Val_model_heatmap(config['model'], device=device)
    val_agent.loadModel()

    tracker = PointTracker(max_length=2, nn_thresh=val_agent.nn_thresh)

    count = 0

    for i, sample in tqdm(enumerate(test_loader)):
        img_0, img_1 = sample['image'], sample['warped_image']

        def get_pts_desc_from_agent(val_agent, img, device='gpu'):

            heatmap_batch = val_agent.run(img)
            pts = val_agent.heatmap_to_pts()
            if subpixel:
                pts = val_agent.soft_argmax_points(pts, patch_size=patch_size)

            desc_sparse = val_agent.desc_to_sparseDesc()

            outs = {'pts': pts[0], 'desc': desc_sparse[0]}
            return outs

        def transpose_np_dict(outs):
            for entry in list(outs):
                outs[entry] = outs[entry].transpose()

        outs = get_pts_desc_from_agent(val_agent, img_0, device=device)
        pts, desc = outs['pts'], outs['desc']

        if outputMatches == True:
            tracker.update(pts, desc)

        pred = {'image': squeezeToNumpy(img_0)}
        pred.update({'prob': pts.transpose(), 'desc': desc.transpose()})

        outs = get_pts_desc_from_agent(val_agent, img_1, device=device)
        pts, desc = outs['pts'], outs['desc']

        if outputMatches == True:
            tracker.update(pts, desc)

        pred.update({'warped_image': squeezeToNumpy(img_1)})
        pred.update({'warped_prob': pts.transpose(),
                     'warped_desc': desc.transpose(),
                     'homography': squeezeToNumpy(sample['homography'])})

        if outputMatches == True:
            matches = tracker.get_matches()
            print('matches: ', matches.transpose().shape)
            pred.update({'matches': matches.transpose()})
        print('pts: ', pts.shape, ', desc: ', desc.shape)

        tracker.clear_desc()

        filename = str(count)
        path = Path(save_output, '{}.npz'.format(filename))
        np.savez_compressed(path, **pred)
        count += 1
    print('output pairs: ', count)


@paddle.no_grad()
def export_detector_homoAdapt_gpu(config, output_dir, args):

    from utils.utils import pltImshow
    from utils.utils import saveImg
    from utils.draw import draw_keypoints

    task = config['data']['dataset']
    export_task = config['data']['export_folder']
    device = 'gpu'

    paddle.device.set_device(device)

    logging.info('train on device: %s', device)

    with open(os.path.join(output_dir, 'config.yml'), 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    writer = LogWriter(getWriterPath(task=args.command,
                                     exper_name=args.exper_name, date=True))

    nms_dist = config['model']['nms']
    top_k = config['model']['top_k']
    homoAdapt_iter = config['data']['homography_adaptation']['num']
    conf_thresh = config['model']['detection_threshold']
    nn_thresh = 0.7
    outputMatches = True
    count = 0
    max_length = 5
    output_images = args.outputImg
    check_exist = True

    save_path = Path(output_dir)
    save_output = save_path
    save_output = save_output / 'predictions' / export_task
    save_path = save_path / 'checkpoints'
    logging.info('=> will save everything to {}'.format(save_path))
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(save_output, exist_ok=True)

    from utils.loader import dataLoader_test as dataLoader

    data = dataLoader(config, dataset=task, export_task=export_task)
    test_set, test_loader = data['test_set'], data['test_loader']

    try:
        path = config['pretrained']
        print('==> Loading pre-trained network.')
        print('path: ', path)

        fe = SuperPointFrontend_torch(config=config,
                                      weights_path=path,
                                      nms_dist=nms_dist,
                                      conf_thresh=conf_thresh,
                                      nn_thresh=nn_thresh,
                                      cuda=False, device=device)
        print('==> Successfully loaded pre-trained network.')

        fe.net_parallel()
        print(path)
        save_file = save_output / 'export.txt'
        with open(save_file, 'a') as myfile:
            myfile.write('load model: ' + path + '\n')
    except Exception:
        print(f'load model: {path} failed! ')
        raise

    def load_as_float(path):
        return imread(path).astype(np.float32) / 255

    tracker = PointTracker(max_length, nn_thresh=fe.nn_thresh)
    with open(save_file, 'a') as myfile:
        myfile.write('homography adaptation: ' + str(homoAdapt_iter) + '\n')

    for i, sample in tqdm(enumerate(test_loader)):
        img, mask_2D = sample['image'], sample['valid_mask']
        img = img.transpose(0, 1)
        img_2D = sample['image_2D'].numpy().squeeze()
        mask_2D = mask_2D.transpose(0, 1)

        inv_homographies, homographies = sample['homographies'], sample['inv_homographies']
        img, mask_2D, homographies, inv_homographies = img, \
                                                       mask_2D, \
                                                       homographies,\
                                                       inv_homographies

        name = sample['name'][0]
        logging.info(f'name: {name}')
        if check_exist:
            p = Path(save_output, '{}.npz'.format(name))
            if p.exists():
                logging.info('file %s exists. skip the sample.', name)
                continue

        heatmap = fe.run(img, onlyHeatmap=True, train=False)
        outputs = combine_heatmap(heatmap, inv_homographies, mask_2D,
                                  device=device)
        pts = fe.getPtsFromHeatmap(outputs.detach().cpu().squeeze())

        if config['model']['subpixel']['enable']:
            fe.heatmap = outputs
            print('outputs: ', outputs.shape)
            print('pts: ', pts.shape)
            pts = fe.soft_argmax_points([pts])
            pts = pts[0]

        pts = pts.transpose()
        print('total points: ', pts.shape)
        print('pts: ', pts[:5])
        if top_k:
            if pts.shape[0] > top_k:
                pts = pts[:top_k, :]
                print('topK filter: ', pts.shape)

        pred = {}
        pred.update({'pts': pts})

        filename = str(name)
        if task == 'Kitti' or 'Kitti_inh':
            scene_name = sample['scene_name'][0]
            os.makedirs(Path(save_output, scene_name), exist_ok=True)

        path = Path(save_output, '{}.npz'.format(filename))
        np.savez_compressed(path, **pred)

        if output_images:
            img_pts = draw_keypoints(img_2D * 255, pts.transpose())
            f = save_output / (str(count) + '.png')
            if task == 'Coco' or 'Kitti':
                f = save_output / (name + '.png')
            saveImg(img_pts, str(f))
        count += 1

    print('output pseudo ground truth: ', count)
    save_file = save_output / 'export.txt'
    with open(save_file, 'a') as myfile:
        myfile.write('Homography adaptation: ' + str(homoAdapt_iter) + '\n')
        myfile.write('output pairs: ' + str(count) + '\n')
    pass


if __name__ == '__main__':
    paddle.set_default_dtype('float32')
    device = paddle.device.set_device('gpu')

    logging.basicConfig(format='[%(asctime)s %(levelname)s] %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command')

    p_train = subparsers.add_parser('export_descriptor')
    p_train.add_argument('config', type=str)
    p_train.add_argument('exper_name', type=str)
    p_train.add_argument('--correspondence', action='store_true')
    p_train.add_argument('--eval', action='store_true')
    p_train.add_argument('--debug', action='store_true', default=False,
                         help='turn on debuging mode')
    p_train.set_defaults(func=export_descriptor)

    p_train = subparsers.add_parser('export_detector_homoAdapt')
    p_train.add_argument('config', type=str)
    p_train.add_argument('exper_name', type=str)
    p_train.add_argument('--eval', action='store_true')
    p_train.add_argument('--outputImg', action='store_true',
                         help='output image for visualization')
    p_train.add_argument('--debug', action='store_true', default=False,
                         help='turn on debuging mode')
    p_train.set_defaults(func=export_detector_homoAdapt_gpu)

    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    print('check config!! ', config)

    output_dir = os.path.join(EXPER_PATH, args.exper_name)
    os.makedirs(output_dir, exist_ok=True)

    logging.info('Running command {}'.format(args.command.upper()))
    args.func(config, output_dir, args)
