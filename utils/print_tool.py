"""tools to print object shape or type
"""
import logging

def print_config(config, file=None):
    print('=' * 10, ' important config: ', '=' * 10, file=file)
    for item in list(config):
        print(item, ': ', config[item], file=file)
    print('=' * 32)


def print_dict_attr(dictionary, attr=None, file=None):
    for item in list(dictionary):
        d = dictionary[item]
        if attr == None:
            print(item, ': ', d, file=file)
        elif hasattr(d, attr):
            print(item, ': ', getattr(d, attr), file=file)
        else:
            print(item, ': ', len(d), file=file)


def datasize(train_loader, config, tag='train'):
    logging.info('== %s split size %d in %d batches' % (tag,
                                                        len(train_loader) * config['model']['batch_size'],
                                                        len(train_loader)))
    pass
