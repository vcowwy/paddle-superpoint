"""Training script
This is the training script for superpoint detector and descriptor.
"""
import argparse
import yaml
import os
import logging

import paddle
import paddle.optimizer
import paddle.io

from visualdl import LogWriter

from utils.utils import getWriterPath
from settings import EXPER_PATH

## loaders: data, model, pretrained model
from utils.loader import dataLoader, modelLoader, pretrainedLoader
from utils.logging import *
from utils.loader import get_save_path


def datasize(train_loader, config, tag='train'):
    logging.info('== %s split size %d in %d batches' % (tag, len(
        train_loader) * config['model']['batch_size'], len(train_loader)))
    pass


def train_base(config, output_dir, args):
    return train_joint(config, output_dir, args)
    pass


def train_joint(config, output_dir, args):
    assert 'train_iter' in config

    paddle.set_default_dtype('float32')
    task = config['data']['dataset']

    device = paddle.set_device('gpu')

    logging.info('train on device: %s', device)

    with open(os.path.join(output_dir, 'config.yml'), 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    writer = LogWriter(getWriterPath(task=args.command,
                                     exper_name=args.exper_name, date=True))

    save_path = get_save_path(output_dir)

    data = dataLoader(config, dataset=task, warp_input=True)
    train_loader, val_loader = data['train_loader'], data['val_loader']

    datasize(train_loader, config, tag='train')
    datasize(val_loader, config, tag='val')

    from utils.loader import get_module
    train_model_frontend = get_module('', config['front_end_model'])

    train_agent = train_model_frontend(config, save_path=save_path, device=device)

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


if __name__ == '__main__':
    paddle.set_default_dtype('float32')
    device = paddle.device.set_device('gpu')

    logging.basicConfig(format='[%(asctime)s %(levelname)s] %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command')

    p_train = subparsers.add_parser('train_base')
    p_train.add_argument('config', type=str)
    p_train.add_argument('exper_name', type=str)
    p_train.add_argument('--eval', action='store_true')
    p_train.add_argument('--debug', action='store_true', default=False,
                         help='turn on debuging mode')
    p_train.set_defaults(func=train_base)

    p_train = subparsers.add_parser('train_joint')
    p_train.add_argument('config', type=str)
    p_train.add_argument('exper_name', type=str)
    p_train.add_argument('--eval', action='store_true')
    p_train.add_argument('--debug', action='store_true', default=False,
                         help='turn on debuging mode')
    p_train.set_defaults(func=train_joint)

    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(format='[%(asctime)s %(levelname)s] %(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S', level=logging.DEBUG)

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    output_dir = os.path.join(EXPER_PATH, args.exper_name)
    os.makedirs(output_dir, exist_ok=True)

    logging.info('Running command {}'.format(args.command.upper()))
    args.func(config, output_dir, args)
