"""

The purpose of this file is to perform data augmentation for images
and lists of pixel positions in them.

- For operations on the images, we can use functions optimized 
for image data.

- For operations on a list of pixel indices, we need a matching
implementation.

"""
from PIL import Image
from PIL import ImageOps
import numpy as np
import random

import paddle

from utils.t2p import LongTensor


def random_image_and_indices_mutation(images, uv_pixel_positions):
    if random.random() < 0.5:
        return images, uv_pixel_positions
    else:
        mutated_images, mutated_uv_pixel_positions = flip_vertical(images, uv_pixel_positions)
        mutated_images, mutated_uv_pixel_positions = flip_horizontal(mutated_images, mutated_uv_pixel_positions)

        return mutated_images, mutated_uv_pixel_positions


def flip_vertical(images, uv_pixel_positions):
    mutated_images = [ImageOps.flip(image) for image in images]
    v_pixel_positions = uv_pixel_positions[1]
    mutated_v_pixel_positions = image.height - 1 - v_pixel_positions
    mutated_uv_pixel_positions = uv_pixel_positions[0], mutated_v_pixel_positions
    return mutated_images, mutated_uv_pixel_positions


def flip_horizontal(images, uv_pixel_positions):
    mutated_images = [ImageOps.mirror(image) for image in images]
    u_pixel_positions = uv_pixel_positions[0]
    mutated_u_pixel_positions = image.width - 1 - u_pixel_positions
    mutated_uv_pixel_positions = mutated_u_pixel_positions, uv_pixel_positions[1]
    return mutated_images, mutated_uv_pixel_positions


def random_domain_randomize_background(image_rgb, image_mask):
    if random.random() < 0.5:
        return image_rgb
    else:
        return domain_randomize_background(image_rgb, image_mask)


def domain_randomize_background(image_rgb, image_mask):
    image_rgb_numpy = np.asarray(image_rgb)

    image_mask_numpy = np.asarray(image_mask)

    three_channel_mask = np.zeros_like(image_rgb_numpy)
    three_channel_mask[:, :, 0] = three_channel_mask[:, :, 1] = three_channel_mask[:, :, 2] = image_mask
    image_rgb_numpy = image_rgb_numpy * three_channel_mask

    three_channel_mask_complement = np.ones_like(three_channel_mask) - three_channel_mask
    random_rgb_image = get_random_image(image_rgb_numpy.shape)
    random_rgb_background = three_channel_mask_complement * random_rgb_image

    domain_randomized_image_rgb = image_rgb_numpy + random_rgb_background
    return Image.fromarray(domain_randomized_image_rgb)


def get_random_image(shape):
    if random.random() < 0.5:
        rand_image = get_random_solid_color_image(shape)
    else:
        rgb1 = get_random_solid_color_image(shape)
        rgb2 = get_random_solid_color_image(shape)
        vertical = bool(np.random.uniform() > 0.5)
        rand_image = get_gradient_image(rgb1, rgb2, vertical=vertical)
    if random.random() < 0.5:
        return rand_image
    else:
        return add_noise(rand_image)


def get_random_rgb():
    return np.array(np.random.uniform(size=3) * 255, dtype=np.uint8)


def get_random_solid_color_image(shape):
    return np.ones(shape, dtype=np.uint8) * get_random_rgb()


def get_random_entire_image(shape, max_pixel_uint8):
    return np.array(np.random.uniform(size=shape) * max_pixel_uint8, dtype=np.uint8)


def get_gradient_image(rgb1, rgb2, vertical):
    bitmap = np.zeros_like(rgb1)
    h, w = rgb1.shape[0], rgb1.shape[1]
    if vertical:
        p = np.tile(np.linspace(0, 1, h)[:, None], (1, w))
    else:
        p = np.tile(np.linspace(0, 1, w), (h, 1))

    for i in range(3):
        bitmap[:, :, i] = rgb2[:, :, i] * p + rgb1[:, :, i] * (1.0 - p)
    return bitmap


def add_noise(rgb_image):
    max_noise_to_add_or_subtract = 50
    return rgb_image + get_random_entire_image(rgb_image.shape,
                                               max_noise_to_add_or_subtract) - get_random_entire_image(rgb_image.shape,
                                                                                                       max_noise_to_add_or_subtract)


def merge_images_with_occlusions(image_a, image_b, mask_a, mask_b, matches_pair_a, matches_pair_b):
    if random.random() < 0.5:
        foreground = 'B'
        background_image, background_mask, background_matches_pair = (image_a, mask_a, matches_pair_a)
        foreground_image, foreground_mask, foreground_matches_pair = (image_b, mask_b, matches_pair_b)
    else:
        foreground = 'A'
        background_image, background_mask, background_matches_pair = (image_b, mask_b, matches_pair_b)
        foreground_image, foreground_mask, foreground_matches_pair = (image_a, mask_a, matches_pair_a)

    foreground_image_numpy = np.asarray(foreground_image)
    foreground_mask_numpy = np.asarray(foreground_mask)
    three_channel_mask = np.zeros_like(foreground_image_numpy)
    three_channel_mask[:, :, 0] = three_channel_mask[:, :, 1] = three_channel_mask[:, :, 2] = foreground_mask
    foreground_image_numpy = foreground_image_numpy * three_channel_mask

    background_image_numpy = np.asarray(background_image)
    three_channel_mask_complement = np.ones_like(three_channel_mask) - three_channel_mask
    background_image_numpy = (three_channel_mask_complement * background_image_numpy)

    merged_image_numpy = foreground_image_numpy + background_image_numpy

    background_matches_pair = prune_matches_if_occluded(foreground_mask_numpy, background_matches_pair)
    if foreground == 'A':
        matches_a = foreground_matches_pair[0]
        associated_matches_a = foreground_matches_pair[1]
        matches_b = background_matches_pair[0]
        associated_matches_b = background_matches_pair[1]
    elif foreground == 'B':
        matches_a = background_matches_pair[0]
        associated_matches_a = background_matches_pair[1]
        matches_b = foreground_matches_pair[0]
        associated_matches_b = foreground_matches_pair[1]
    else:
        raise ValueError('Should not be here?')

    merged_masked_numpy = foreground_mask_numpy + np.asarray(background_mask)
    merged_masked_numpy = merged_masked_numpy.clip(0, 1)
    return (Image.fromarray(merged_image_numpy), merged_masked_numpy, matches_a, associated_matches_a, matches_b, associated_matches_b)


def prune_matches_if_occluded(foreground_mask_numpy, background_matches_pair):
    background_matches_a = background_matches_pair[0]
    background_matches_b = background_matches_pair[1]

    idxs_to_keep = []

    for i in range(len(background_matches_a[0])):
        u = background_matches_a[0][i]
        v = background_matches_a[1][i]

        if foreground_mask_numpy[v, u] == 0:
            idxs_to_keep.append(i)

    if len(idxs_to_keep) == 0:
        return None, None

    idxs_to_keep = LongTensor(idxs_to_keep)
    background_matches_a = (paddle.index_select(background_matches_a[0], 0, idxs_to_keep), paddle.index_select(background_matches_a[1], 0, idxs_to_keep))
    background_matches_b = (paddle.index_select(background_matches_b[0], 0, idxs_to_keep), paddle.index_select(background_matches_b[1], 0, idxs_to_keep))

    return background_matches_a, background_matches_b


def merge_matches(matches_one, matches_two):
    concatenated_u = paddle.concat((matches_one[0], matches_two[0]))
    concatenated_v = paddle.concat((matches_one[1], matches_two[1]))
    return concatenated_u, concatenated_v
