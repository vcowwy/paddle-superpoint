"""Sample homography matrices
# mimic the function from tensorflow
# very tricky. Need to be careful for using the parameters.
"""
from math import pi
import cv2
import numpy as np

from utils.tools import dict_update


def sample_homography_np(shape, shift=0, perspective=True, scaling=True,
                         rotation=True, translation=True, n_scales=5, n_angles=25,
                         scaling_amplitude=0.1, perspective_amplitude_x=0.1,
                         perspective_amplitude_y=0.1, patch_ratio=0.5, max_angle=pi / 2,
                         allow_artifacts=False, translation_overflow=0.0):

    pts1 = np.stack([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]], axis=0)

    margin = (1 - patch_ratio) / 2
    pts2 = margin + np.array([[0, 0], [0, patch_ratio],
                              [patch_ratio, patch_ratio],
                              [patch_ratio, 0]])

    from numpy.random import normal
    from numpy.random import uniform
    from scipy.stats import truncnorm

    std_trunc = 2

    if perspective:
        if not allow_artifacts:
            perspective_amplitude_x = min(perspective_amplitude_x, margin)
            perspective_amplitude_y = min(perspective_amplitude_y, margin)

        perspective_displacement = truncnorm(-1 * std_trunc, std_trunc, loc=0, scale=perspective_amplitude_y / 2).rvs(1)

        h_displacement_left = truncnorm(-1 * std_trunc, std_trunc, loc=0, scale=perspective_amplitude_x / 2).rvs(1)

        h_displacement_right = truncnorm(-1 * std_trunc, std_trunc, loc=0, scale=perspective_amplitude_x / 2).rvs(1)

        pts2 += np.array([[h_displacement_left, perspective_displacement],
                          [h_displacement_left, -perspective_displacement],
                          [h_displacement_right, perspective_displacement],
                          [h_displacement_right, -perspective_displacement]]).squeeze()
    if scaling:
        scales = truncnorm(-1 * std_trunc, std_trunc, loc=1, scale=scaling_amplitude / 2).rvs(n_scales)
        scales = np.concatenate((np.array([1]), scales), axis=0)

        center = np.mean(pts2, axis=0, keepdims=True)
        scaled = (pts2 - center)[np.newaxis, :, :] * scales[:, np.newaxis, np.newaxis] + center
        if allow_artifacts:
            valid = np.arange(n_scales)
        else:
            valid = (scaled >= 0.0) * (scaled < 1.0)
            valid = valid.prod(axis=1).prod(axis=1)
            valid = np.where(valid)[0]
        idx = valid[np.random.randint(valid.shape[0], size=1)].squeeze().astype(int)
        pts2 = scaled[idx, :, :]

    if translation:
        t_min, t_max = np.min(pts2, axis=0), np.min(1 - pts2, axis=0)
        if allow_artifacts:
            t_min += translation_overflow
            t_max += translation_overflow
        pts2 += np.array([uniform(-t_min[0], t_max[0], 1), uniform(-t_min[1], t_max[1], 1)]).T

    if rotation:
        angles = np.linspace(-max_angle, max_angle, num=n_angles)
        angles = np.concatenate((angles, np.array([0.0])), axis=0)
        center = np.mean(pts2, axis=0, keepdims=True)
        rot_mat = np.reshape(np.stack([np.cos(angles), -np.sin(angles),
                                       np.sin(angles), np.cos(angles)], axis=1), [-1, 2, 2])
        rotated = np.matmul((pts2 - center)[np.newaxis, :, :], rot_mat) + center
        if allow_artifacts:
            valid = np.arange(n_angles)
        else:
            valid = (rotated >= 0.0) * (rotated < 1.0)
            valid = valid.prod(axis=1).prod(axis=1)
            valid = np.where(valid)[0]
        idx = valid[np.random.randint(valid.shape[0], size=1)].squeeze().astype(int)
        pts2 = rotated[idx, :, :]

    shape = shape[::-1]
    pts1 *= shape[np.newaxis, :]
    pts2 *= shape[np.newaxis, :]

    def ax(p, q):
        return [p[0], p[1], 1, 0, 0, 0, -p[0] * q[0], -p[1] * q[0]]

    def ay(p, q):
        return [0, 0, 0, p[0], p[1], 1, -p[0] * q[1], -p[1] * q[1]]
    homography = cv2.getPerspectiveTransform(np.float32(pts1 + shift), np.float32(pts2 + shift))
    return homography

"""
def sample_homography(shape, perspective=True, scaling=True, rotation=True,
                      translation=True, n_scales=5, n_angles=25, scaling_amplitude=0.1,
                      perspective_amplitude_x=0.1, perspective_amplitude_y=0.1,
                      patch_ratio=0.5, max_angle=pi / 2, allow_artifacts=False, translation_overflow=0.0):

    import tensorflow as tf
    from tensorflow.contrib.image import transform as H_transform

    pts1 = tf.stack([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]], axis=0)
    margin = (1 - patch_ratio) / 2
    pts2 = margin + tf.constant([[0, 0], [0, patch_ratio],
                                 [patch_ratio, patch_ratio],
                                 [patch_ratio, 0]], tf.float32)
    if perspective:
        if not allow_artifacts:
            perspective_amplitude_x = min(perspective_amplitude_x, margin)
            perspective_amplitude_y = min(perspective_amplitude_y, margin)
        perspective_displacement = tf.truncated_normal([1], 0.0,  perspective_amplitude_y / 2)
        h_displacement_left = tf.truncated_normal([1], 0.0,  perspective_amplitude_x / 2)
        h_displacement_right = tf.truncated_normal([1], 0.0,  perspective_amplitude_x / 2)
        pts2 += tf.stack([tf.concat([h_displacement_left, perspective_displacement], 0),
                          tf.concat([h_displacement_left,  -perspective_displacement], 0),
                          tf.concat([h_displacement_right, perspective_displacement], 0),
                          tf.concat([h_displacement_right, -perspective_displacement], 0)])

    if scaling:
        scales = tf.concat([[1.0], tf.truncated_normal([n_scales], 1, scaling_amplitude / 2)], 0)
        center = tf.reduce_mean(pts2, axis=0, keepdims=True)
        scaled = tf.expand_dims(pts2 - center, axis=0) * tf.expand_dims(tf.expand_dims(scales, 1), 1) + center
        if allow_artifacts:
            valid = tf.range(n_scales)
        else:
            valid = tf.where(tf.reduce_all((scaled >= 0.0) & (scaled < 1.0), [1, 2]))[:, 0]
        idx = valid[tf.random_uniform((), maxval=tf.shape(valid)[0], dtype=tf.int32)]
        pts2 = scaled[idx]

    if translation:
        t_min, t_max = tf.reduce_min(pts2, axis=0), tf.reduce_min(1 - pts2, axis=0)
        if allow_artifacts:
            t_min += translation_overflow
            t_max += translation_overflow
        pts2 += tf.expand_dims(tf.stack([tf.random_uniform((), -t_min[0], t_max[0]),
                                         tf.random_uniform((), -t_min[1], t_max[1])]),
                               axis=0)

    if rotation:
        angles = tf.lin_space(tf.constant(-max_angle), tf.constant(max_angle), n_angles)
        angles = tf.concat([angles, [0.0]], axis=0)
        center = tf.reduce_mean(pts2, axis=0, keepdims=True)
        rot_mat = tf.reshape(tf.stack([tf.cos(angles), -tf.sin(angles),
                                       tf.sin(angles), tf.cos(angles)], axis=1), [-1, 2, 2])
        rotated = tf.matmul(tf.tile(tf.expand_dims(pts2 - center, axis=0),
                                    [n_angles + 1, 1, 1]), rot_mat) + center
        if allow_artifacts:
            valid = tf.range(n_angles)
        else:
            valid = tf.where(tf.reduce_all((rotated >= 0.0) & (rotated < 1.0),
                                           axis=[1, 2]))[:, 0]
        idx = valid[tf.random_uniform((), maxval=tf.shape(valid)[0], dtype=tf.int32)]
        pts2 = rotated[idx]

    shape = tf.to_float(shape[::-1])
    pts1 *= tf.expand_dims(shape, axis=0)
    pts2 *= tf.expand_dims(shape, axis=0)

    def ax(p, q):
        return [p[0], p[1], 1, 0, 0, 0, -p[0] * q[0], -p[1] * q[0]]

    def ay(p, q):
        return [0, 0, 0, p[0], p[1], 1, -p[0] * q[1], -p[1] * q[1]]

    a_mat = tf.stack([f(pts1[i], pts2[i]) for i in range(4) for f in (ax,
        ay)], axis=0)
    p_mat = tf.transpose(tf.stack(
        [[pts2[i][j] for i in range(4)
          for j in range(2)]], axis=0))
    homography = tf.transpose(tf.matrix_solve_ls(a_mat, p_mat, fast=True))
    sess = tf.Session()
    with sess.as_default():
        homography = homography.eval()
    homography = np.concatenate((homography, np.array([[1]])), axis=1)
    homography = homography.reshape(3, 3)
    return homography
"""

import paddle


def scale_homography_torch(H, shape, shift=(-1, -1), dtype=paddle.float32):
    height, width = shape[0], shape[1]
    trans = paddle.to_tensor([[2.0 / width, 0.0, shift[0]], [0.0, 2.0 /height, shift[1]], [0.0, 0.0, 1.0]], dtype=dtype)

    H_tf = paddle.matmul(paddle.matmul(paddle.inverse(trans), H), trans)
    return H_tf


def scale_homography(H, shape, shift=(-1, -1)):
    height, width = shape[0], shape[1]
    trans = np.array([[2.0 / width, 0.0, shift[0]], [0.0, 2.0 / height, shift[1]], [0.0, 0.0, 1.0]])
    H_tf = np.linalg.inv(trans) @ H @ trans
    return H_tf
