"""many loaders
# loader for model, dataset, testing dataset
"""
import os
import logging
from pathlib import Path
import numpy as np

import paddle
import paddle.optimizer

from utils.utils import tensor2array
from utils.utils import save_checkpoint
from utils.utils import load_checkpoint
from utils.utils import save_path_formatter
from utils.t2p import IntTensor


def get_save_path(output_dir):
    save_path = Path(output_dir)
    save_path = save_path / 'checkpoints'
    logging.info('=> will save everything to {}'.format(save_path))
    os.makedirs(save_path, exist_ok=True)
    return save_path


def worker_init_fn(worker_id):
    base_seed = paddle.to_tensor(data=1, dtype=paddle.int8).random_().item()
    np.random.seed(base_seed + worker_id)


def dataLoader(config, dataset='syn', warp_input=False, train=True, val=True):
    import paddle.vision.transforms as transforms

    training_params = config.get('training', {})
    workers_train = training_params.get('workers_train', 1)
    workers_val = training_params.get('workers_val', 1)

    logging.info(f'workers_train: {workers_train}, workers_val: {workers_val}')
    data_transforms = {
        'train': transforms.Compose([transforms.ToTensor(), ]),
        'val': transforms.Compose([transforms.ToTensor(), ])
    }

    Dataset = get_module('datasets', dataset)
    print(f'dataset: {dataset}')

    train_set = Dataset(
        transform=data_transforms['train'],
        task='train',
        **config['data'])

    train_loader = paddle.io.DataLoader(
        train_set,
        batch_size=config['model']['batch_size'],
        shuffle=True,
        num_workers=workers_train,
        worker_init_fn=worker_init_fn)
    val_set = Dataset(
        transform=data_transforms['train'],
        task='val', **
        config['data'])
    val_loader = paddle.io.DataLoader(
        val_set,
        batch_size=config['model']['eval_batch_size'],
        shuffle=True,
        num_workers=workers_val,
        worker_init_fn=worker_init_fn)
    return {'train_loader': train_loader, 'val_loader': val_loader,
            'train_set': train_set, 'val_set': val_set}


def dataLoader_test(config, dataset='syn', warp_input=False, export_task='train'):

    import paddle.vision.transforms as transforms

    training_params = config.get('training', {})
    workers_test = training_params.get('workers_test', 1)
    logging.info(f'workers_test: {workers_test}')

    data_transforms = {'test': transforms.Compose([transforms.ToTensor(), ])}
    test_loader = None
    if dataset == 'syn':
        from datasets.SyntheticDataset_gaussian import SyntheticDataset_gaussian
        #from datasets.SyntheticDataset import SyntheticDataset
        test_set = SyntheticDataset_gaussian(transform=data_transforms['test'],
                                    train=False,
                                    warp_input=warp_input,
                                    getPts=True, seed=1,
                                    **config['data'])
    elif dataset == 'hpatches':
        from datasets.patches_dataset import PatchesDataset
        if config['data']['preprocessing']['resize']:
            size = config['data']['preprocessing']['resize']
        test_set = PatchesDataset(
            transform=data_transforms['test'],
            **config['data'])

        test_loader = paddle.io.DataLoader(test_set,
                                           batch_size=1,
                                           shuffle=False,
                                           num_workers=workers_test,
                                           worker_init_fn=None)
    else:
        logging.info(f'load dataset from : {dataset}')
        Dataset = get_module('datasets', dataset)
        test_set = Dataset(export=True, task=export_task, **config['data'])
        test_loader = paddle.io.DataLoader(
            test_set,
            batch_size=1,
            shuffle=False,
            num_workers=workers_test,
            worker_init_fn=worker_init_fn)

    return {'test_set': test_set, 'test_loader': test_loader}


def get_module(path, name):
    import importlib
    if path == '':
        mod = importlib.import_module(name)
    else:
        mod = importlib.import_module('{}.{}'.format(path, name))
    return getattr(mod, name)


def get_model(name):
    mod = __import__('models.{}'.format(name), fromlist=[''])
    return getattr(mod, name)


def modelLoader(model='SuperPointNet', **options):
    logging.info('=> creating model: %s', model)
    net = get_model(model)
    net = net(**options)
    return net


def pretrainedLoader(net, optimizer, epoch, path, mode='full', full_path=False):
    if full_path == True:
        checkpoint = paddle.load(path)
    else:
        checkpoint = load_checkpoint(path)

    if mode == 'full':
        net.load_dict(checkpoint['model_state_dict'])
        optimizer.load_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['n_iter']
    else:
        net.load_dict(checkpoint)
    return net, optimizer, epoch


if __name__ == '__main__':
    net = modelLoader(model='SuperPointNet')
