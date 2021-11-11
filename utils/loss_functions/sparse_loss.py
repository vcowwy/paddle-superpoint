import numpy as np

import paddle

import utils.correspondence_tools.correspondence_finder as correspondence_finder
from utils.homographies import scale_homography_torch
from utils.loss_functions.pixelwise_contrastive_loss import PixelwiseContrastiveLoss


def get_coor_cells(Hc, Wc, cell_size, device='gpu', uv=False):
    coor_cells = paddle.stack(paddle.meshgrid(paddle.arange(Hc), paddle.arange(Wc)), axis=2)
    coor_cells = paddle.to_tensor(coor_cells, dtype=paddle.float32)
    coor_cells = paddle.reshape(coor_cells, shape=[-1, 2])
    if uv:
        coor_cells = paddle.stack((coor_cells[:, (1)], coor_cells[:, (0)]), axis=1)
    return coor_cells


def warp_coor_cells_with_homographies(coor_cells, homographies, uv=False,
    device='gpu'):
    from utils.utils import warp_points
    warped_coor_cells = coor_cells
    if uv == False:
        warped_coor_cells = paddle.stack((warped_coor_cells[:, (1)], warped_coor_cells[:, (0)]),
                                         axis=1)
    warped_coor_cells = warp_points(warped_coor_cells, homographies, device)
    if uv == False:
        warped_coor_cells = paddle.stack((warped_coor_cells[:, :, (1)],  warped_coor_cells[:, :, (0)]),
                                         axis=2)
    return warped_coor_cells


def create_non_matches(uv_a, uv_b_non_matches, multiplier):
    uv_a_long = (paddle.reshape(paddle.t(uv_a[0].repeat(multiplier, 1)).contiguous(), shape=[-1, 1]),
                 paddle.reshape(paddle.t(uv_a[1].repeat(multiplier, 1)).contiguous(), shape=[-1, 1]))

    uv_b_non_matches_long = paddle.reshape(uv_b_non_matches[0], shape=[-1, 1]), paddle.reshape(uv_b_non_matches[1], shape=[-1, 1])
    return uv_a_long, uv_b_non_matches_long


def descriptor_loss_sparse(descriptors, descriptors_warped, homographies,
                           mask_valid=None, cell_size=8, device='gpu', descriptor_dist=4,
                           lamda_d=250, num_matching_attempts=1000,
                           num_masked_non_matches_per_match=10,
                           dist='cos', method='1d', **config):

    def uv_to_tuple(uv):
        return uv[:, 0], uv[:, 1]

    def tuple_to_uv(uv_tuple):
        return paddle.stack([uv_tuple[0], uv_tuple[1]])

    def tuple_to_1d(uv_tuple, W, uv=True):
        if uv:
            return uv_tuple[0] + uv_tuple[1] * W
        else:
            return uv_tuple[0] * W + uv_tuple[1]

    def uv_to_1d(points, W, uv=True):
        if uv:
            return points[..., 0] + points[..., 1] * W
        else:
            return points[..., 0] * W + points[..., 1]

    def get_match_loss(image_a_pred, image_b_pred, matches_a, matches_b,
                       dist='cos', method='1d'):
        match_loss, matches_a_descriptors, matches_b_descriptors = (
            PixelwiseContrastiveLoss.match_loss(image_a_pred, image_b_pred,
                                                matches_a, matches_b, dist=dist, method=method))
        return match_loss

    def get_non_matches_corr(img_b_shape, uv_a, uv_b_matches,
                             num_masked_non_matches_per_match=10, device='gpu'):
        uv_b_matches = uv_b_matches.squeeze()
        uv_b_matches_tuple = uv_to_tuple(uv_b_matches)
        uv_b_non_matches_tuple = (correspondence_finder.create_non_correspondences(
            uv_b_matches_tuple, img_b_shape,
            num_non_matches_per_match=num_masked_non_matches_per_match,
            img_b_mask=None))
        uv_a_tuple, uv_b_non_matches_tuple = create_non_matches(uv_to_tuple
                                                                (uv_a), uv_b_non_matches_tuple,
                                                                num_masked_non_matches_per_match)
        return uv_a_tuple, uv_b_non_matches_tuple

    def get_non_match_loss(image_a_pred, image_b_pred, non_matches_a,
                           non_matches_b, dist='cos'):
        (non_match_loss, num_hard_negatives, non_matches_a_descriptors,
         non_matches_b_descriptors) = (PixelwiseContrastiveLoss.non_match_descriptor_loss(
            image_a_pred, image_b_pred,
            paddle.to_tensor(non_matches_a, dtype=paddle.int64).squeeze(),
            paddle.to_tensor(non_matches_b, dtype=paddle.int64).squeeze(),
            M=0.2, invert=True, dist=dist))
        non_match_loss = non_match_loss.sum() / (num_hard_negatives + 1)
        return non_match_loss

    from utils.utils import filter_points
    from utils.utils import crop_or_pad_choice
    from utils.utils import normPts

    Hc, Wc = descriptors.shape[1], descriptors.shape[2]
    img_shape = Hc, Wc

    def descriptor_reshape(descriptors):
        descriptors = paddle.transpose(paddle.reshape(descriptors, shape=[-1, Hc * Wc]), perm=[1, 0]
        descriptors = descriptors.unsqueeze(0)
        return descriptors
    image_a_pred = descriptor_reshape(descriptors)
    image_b_pred = descriptor_reshape(descriptors_warped)
    uv_a = get_coor_cells(Hc, Wc, cell_size, uv=True, device='gpu')
    homographies_H = scale_homography_torch(homographies, img_shape, shift=\
        (-1, -1))
    uv_b_matches = warp_coor_cells_with_homographies(uv_a, homographies_H.
        to('cpu'), uv=True, device='gpu')
    uv_b_matches.round_()
    uv_b_matches = uv_b_matches.squeeze(0)
    uv_b_matches, mask = filter_points(uv_b_matches, paddle.to_tensor([Wc,
        Hc]).to(device='gpu'), return_mask=True)
    uv_a = uv_a[mask]
    shuffle = True
    if not shuffle:
        print('shuffle: ', shuffle)
    choice = crop_or_pad_choice(uv_b_matches.shape[0],
        num_matching_attempts, shuffle=shuffle)
    choice = paddle.to_tensor(choice)
    uv_a = uv_a[choice]
    uv_b_matches = uv_b_matches[choice]
    if method == '2d':
        matches_a = normPts(uv_a, paddle.to_tensor([Wc, Hc], dtype=paddle.float32))
        matches_b = normPts(uv_b_matches, paddle.to_tensor([Wc, Hc], dtype=paddle.float32))
    else:
        matches_a = uv_to_1d(uv_a, Wc)
        matches_b = uv_to_1d(uv_b_matches, Wc)
    if method == '2d':
        match_loss = get_match_loss(descriptors, descriptors_warped,
            matches_a, matches_b, dist=dist, method='2d')
    else:
        match_loss = get_match_loss(image_a_pred, image_b_pred, paddle.to_tensor(matches_a, dtype=paddle.int64), paddle.to_tensor(matches_b, dtype=paddle.int64), dist=dist)
    uv_a_tuple, uv_b_non_matches_tuple = get_non_matches_corr(img_shape,
        uv_a, uv_b_matches, num_masked_non_matches_per_match=\
        num_masked_non_matches_per_match)
    non_matches_a = tuple_to_1d(uv_a_tuple, Wc)
    non_matches_b = tuple_to_1d(uv_b_non_matches_tuple, Wc)
    non_match_loss = get_non_match_loss(image_a_pred, image_b_pred,
        non_matches_a, non_matches_b, dist=dist)
    loss = lamda_d * match_loss + non_match_loss
    return loss, lamda_d * match_loss, non_match_loss
    pass



def batch_descriptor_loss_sparse(descriptors, descriptors_warped,
    homographies, **options):
    loss = []
    pos_loss = []
    neg_loss = []
    batch_size = descriptors.shape[0]
    for i in range(batch_size):
        losses = descriptor_loss_sparse(descriptors[i], descriptors_warped[
            i], homographies[i].type(paddle.float32), **options)
        loss.append(losses[0])
        pos_loss.append(losses[1])
        neg_loss.append(losses[2])
    loss, pos_loss, neg_loss = paddle.stack(loss), paddle.stack(pos_loss
        ), paddle.stack(neg_loss)
    return loss.mean(), None, pos_loss.mean(), neg_loss.mean()


if __name__ == '__main__':
    H, W = 240, 320
    cell_size = 8
    Hc, Wc = H // cell_size, W // cell_size
    D = 3
    paddle.seed(0)
    np.random.seed(0)
    batch_size = 2
    device = 'gpu'
    method = '2d'
    num_matching_attempts = 1000
    num_masked_non_matches_per_match = 200
    lamda_d = 1
    homographies = np.identity(3)[np.newaxis, :, :]
    homographies = np.tile(homographies, [batch_size, 1, 1])

    def randomDescriptor():
        descriptors = paddle.to_tensor(np.random.rand(2, D, Hc, Wc) - 0.5,
            dtype=paddle.float32)
        dn = paddle.norm(descriptors, p=2, axis=1)  # Compute the norm.
        descriptors = descriptors.div(paddle.unsqueeze(dn, 1))
        return descriptors
    descriptors = randomDescriptor()
    print('descriptors: ', descriptors.shape)
    descriptors_warped = randomDescriptor()
    descriptor_loss = descriptor_loss_sparse(descriptors[0],
        descriptors_warped[0], paddle.to_tensor(homographies[0], dtype=\
        paddle.float32), method=method)
    print('descriptor_loss: ', descriptor_loss)
    loss = batch_descriptor_loss_sparse(descriptors, descriptors, paddle.
        to_tensor(homographies, dtype=paddle.float32), num_matching_attempts
        =num_matching_attempts, num_masked_non_matches_per_match=\
        num_masked_non_matches_per_match, device=device, lamda_d=lamda_d,
        method=method)
    print('same descriptor_loss (pos should be 0): ', loss)
