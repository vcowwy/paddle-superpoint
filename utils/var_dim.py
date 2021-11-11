"""change the dimension of tensor/ numpy array
"""
import numpy as np


def to3dim(img):
    if img.ndim == 2:
        img = img[:, :, np.newaxis]
    return img


def tensorto4d(inp):
    if len(inp.shape) == 2:
        inp = paddle.reshape(inp, shape=[1, 1, inp.shape[0], inp.shape[1]])
    elif len(inp.shape) == 3:
        inp = paddle.reshape(inp, shape=[1, inp.shape[0], inp.shape[1], inp.shape[2]])
    return inp


def squeezeToNumpy(tensor_arr):
    return tensor_arr.detach().cpu().numpy().squeeze()


def toNumpy(tensor):
    return tensor.detach().cpu().numpy()
