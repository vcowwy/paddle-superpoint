"""util functions
# many old functions, need to clean up
# homography --> homography
# warping
# loss --> delete if useless
"""
import numpy as np
from pathlib import Path
import datetime
import cv2

import paddle
import paddle.nn.functional as F
import paddle.nn as nn

from collections import OrderedDict
from utils.d2s import DepthToSpace
from utils.d2s import SpaceToDepth


def img_overlap(img_r, img_g, img_gray):
    def to_3d(img):
        if len(img.shape) == 2:
            img = img[np.newaxis, ...]
        return img
    img_r, img_g, img_gray = to_3d(img_r), to_3d(img_g), to_3d(img_gray)
    img = np.concatenate((img_gray, img_gray, img_gray), axis=0)
    img[0, :, :] += img_r[0, :, :]
    img[1, :, :] += img_g[0, :, :]
    img[img > 1] = 1
    img[img < 0] = 0
    return img


def thd_img(img, thd=0.015):
    img[img < thd] = 0
    img[img >= thd] = 1
    return img


def toNumpy(tensor):
    return tensor.detach().cpu().numpy()


def save_path_formatter(args, parser):
    print('todo: save path')
    return Path('.')
    pass



def tensor2array(tensor, max_value=255, colormap='rainbow', channel_first=True):
    tensor = tensor.detach().cpu()
    if max_value is None:
        max_value = tensor.max().item()
    if tensor.ndimension() == 2 or tensor.size(0) == 1:
        try:
            import cv2
            if int(cv2.__version__[0]) >= 3:
                color_cvt = cv2.COLOR_BGR2RGB
            else:
                color_cvt = cv2.cv.CV_BGR2RGB
            if colormap == 'rainbow':
                colormap = cv2.COLORMAP_RAINBOW
            elif colormap == 'bone':
                colormap = cv2.COLORMAP_BONE
            array = (255 * tensor.squeeze().numpy() / max_value).clip(0, 255).astype(np.uint8)
            colored_array = cv2.applyColorMap(array, colormap)
            array = cv2.cvtColor(colored_array, color_cvt).astype(np.float32) / 255
        except ImportError:
            if tensor.ndimension() == 2:
                tensor.unsqueeze_(2)
            array = (tensor.expand(tensor.size(0), tensor.size(1), 3).numpy
                () / max_value).clip(0, 1)
        if channel_first:
            array = array.transpose(2, 0, 1)

    elif tensor.ndimension() == 3:
        assert tensor.size(0) == 3
        array = 0.5 + tensor.numpy() * 0.5
        if not channel_first:
            array = array.transpose(1, 2, 0)
    return array


def find_files_with_ext(directory, extension='.npz'):
    list_of_files = []
    import os

    if extension == '.npz':
        for l in os.listdir(directory):
            if l.endswith(extension):
                list_of_files.append(l)
        return list_of_files


def save_checkpoint(save_path, net_state, epoch, filename='checkpoint.pdiparams.tar'):
    file_prefix = ['superPointNet']

    filename = '{}_{}_{}'.format(file_prefix[0], str(epoch), filename)
    paddle.save(net_state, save_path / filename)
    print('save checkpoint to ', filename)
    pass


def load_checkpoint(load_path, filename='checkpoint.pdiparams.tar'):
    file_prefix = ['superPointNet']
    filename = '{}__{}'.format(file_prefix[0], filename)

    checkpoint = paddle.load(load_path / filename)
    print('load checkpoint from ', filename)
    return checkpoint
    pass


def saveLoss(filename, iter, loss, task='train', **options):
    with open(filename, 'a') as myfile:
        myfile.write(task + ' iter: ' + str(iter) + ', ')
        myfile.write('loss: ' + str(loss) + ', ')
        myfile.write(str(options))
        myfile.write('\n')


def saveImg(img, filename):
    import cv2

    cv2.imwrite(filename, img)


def pltImshow(img):
    from matplotlib import pyplot as plt

    plt.imshow(img)
    plt.show()


def loadConfig(filename):
    import yaml

    with open(filename, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def append_csv(file='foo.csv', arr=[]):
    import csv

    with open(file, 'a') as f:
        writer = csv.writer(f)
        if type(arr[0]) is list:
            for a in arr:
                writer.writerow(a)
        else:
            writer.writerow(arr)


def sample_homography(inv_scale=3):
    corner_img = np.array([(-1, -1), (-1, 1), (1, -1), (1, 1)])
    img_offset = corner_img
    corner_map = (np.random.rand(4, 2) - 0.5) * 2 / (inv_scale + 0.01
        ) + img_offset
    matrix = cv2.getPerspectiveTransform(np.float32(corner_img), np.float32
        (corner_map))
    return matrix


def sample_homographies(batch_size=1, scale=10, device='gpu'):
    mat_H = [sample_homography(inv_scale=scale) for i in range(batch_size)]
    mat_H = np.stack(mat_H, axis=0)
    mat_H = paddle.to_tensor(mat_H, dtype=paddle.float32)
    mat_H = mat_H
    mat_H_inv = paddle.stack([paddle.inverse(mat_H[i, :, :]) for i in range(batch_size)])
    mat_H_inv = paddle.to_tensor(mat_H_inv, dtype=paddle.float32)
    mat_H_inv = mat_H_inv
    return mat_H, mat_H_inv


def warpLabels(pnts, homography, H, W):
    import paddle
    from utils.utils import warp_points
    from utils.utils import filter_points

    pnts = paddle.to_tensor(pnts, dtype=paddle.int64)
    homography = paddle.to_tensor(homography, dtype=paddle.float32)
    warped_pnts = warp_points(paddle.stack((pnts[:, (0)], pnts[:, (1)]),
        axis=1), homography)
    warped_pnts = paddle.to_tensor(filter_points(warped_pnts, paddle.to_tensor([W, H])).round(
        ), dtype=paddle.int64)
    return warped_pnts.numpy()


def warp_points_np(points, homographies, device='gpu'):
    batch_size = homographies.shape[0]
    points = np.concatenate((points, np.ones((points.shape[0], 1))), axis=1)
    warped_points = np.tensordot(homographies, points.transpose(), axes=([2
        ], [0]))
    warped_points = warped_points.reshape([batch_size, 3, -1])
    warped_points = warped_points.transpose([0, 2, 1])
    warped_points = warped_points[:, :, :2] / warped_points[:, :, 2:]
    return warped_points


def homography_scaling(homography, H, W):
    trans = np.array([[2.0 / W, 0.0, -1], [0.0, 2.0 / H, -1], [0.0, 0.0, 1.0]])
    homography = np.linalg.inv(trans) @ homography @ trans
    return homography


def homography_scaling_torch(homography, H, W):
    trans = paddle.to_tensor([[2.0 / W, 0.0, -1], [0.0, 2.0 / H, -1], [0.0,
        0.0, 1.0]])
    homography = paddle.matmul(paddle.matmul(trans.inverse(), homography), trans)
    return homography


def filter_points(points, shape, return_mask=False):
    points = paddle.to_tensor(points, dtype=paddle.float32)
    shape = paddle.to_tensor(shape, dtype=paddle.float32)
    mask = paddle.to_tensor(paddle.logical_and(points >= 0, points <= shape - 1), dtype=paddle.int32)
    mask = (paddle.prod(mask, axis=-1) == 1)
    mask = paddle.to_tensor(mask, dtype=paddle.int64)
    if return_mask:
        return points[mask], mask
    return points[mask]


def warp_points(points, homographies, device='gpu'):
    no_batches = len(homographies.shape) == 2
    homographies = homographies.unsqueeze(0) if no_batches else homographies
    batch_size = homographies.shape[0]
    points = paddle.concat((paddle.to_tensor(points, dtype=paddle.float32), paddle.ones([points.shape[0], 1])), axis=1)
    homographies = paddle.reshape(homographies, shape=[batch_size * 3, 3])
    warped_points = paddle.matmul(homographies, paddle.transpose(points, perm=[1, 0]))
    warped_points = paddle.reshape(warped_points, shape=[batch_size, 3, -1])
    warped_points = paddle.transpose(warped_points, perm=[0, 2, 1])
    warped_points = warped_points[:, :, :2] / warped_points[:, :, 2:]
    return warped_points[0, :, :] if no_batches else warped_points


def inv_warp_image_batch(img, mat_homo_inv, device='cpu', mode='bilinear'):
    if len(img.shape) == 2 or len(img.shape) == 3:
        img = paddle.reshape(img, shape=[1, 1, img.shape[0], img.shape[1]])
    if len(mat_homo_inv.shape) == 2:
        mat_homo_inv = paddle.reshape(mat_homo_inv, shape=[1, 3, 3])
    Batch, channel, H, W = img.shape
    coor_cells = paddle.stack(paddle.meshgrid(paddle.linspace(-1, 1, W), paddle.linspace(-1, 1, H)), axis=2)
    coor_cells = coor_cells.transpose(0, 1)
    coor_cells = coor_cells
    coor_cells = coor_cells.contiguous()
    src_pixel_coords = warp_points(paddle.reshape(coor_cells, shape=[-1, 2]), mat_homo_inv, device)
    src_pixel_coords = paddle.reshape(src_pixel_coords, shape=[Batch, H, W, 2])
    src_pixel_coords = paddle.to_tensor(src_pixel_coords, dtype=paddle.float32)
    warped_img = F.grid_sample(img, src_pixel_coords, mode=mode, align_corners=True)
    return warped_img


def inv_warp_image(img, mat_homo_inv, device='gpu', mode='bilinear'):
    warped_img = inv_warp_image_batch(img, mat_homo_inv, device, mode)
    return warped_img.squeeze()


def labels2Dto3D(labels, cell_size, add_dustbin=True):
    batch_size, channel, H, W = labels.shape
    Hc, Wc = H // cell_size, W // cell_size
    space2depth = SpaceToDepth(8)
    labels = space2depth(labels)
    if add_dustbin:
        dustbin = labels.sum(dim=1)
        dustbin = 1 - dustbin
        dustbin[dustbin < 1.0] = 0
        labels = paddle.concat((labels, paddle.reshape(dustbin, shape=[batch_size, 1,
            Hc, Wc])), axis=1)
        dn = labels.sum(dim=1)
        labels = labels.div(paddle.unsqueeze(dn, 1))
    return labels


def labels2Dto3D_flattened(labels, cell_size):
    batch_size, channel, H, W = labels.shape
    Hc, Wc = H // cell_size, W // cell_size
    space2depth = SpaceToDepth(8)
    labels = space2depth(labels)
    dustbin = paddle.ones((batch_size, 1, Hc, Wc)).requires_grad_(False).cuda()
    labels = paddle.concat((labels * 2, paddle.reshape(dustbin, shape=[batch_size, 1,
        Hc, Wc])), axis=1)
    labels = paddle.argmax(labels, axis=1)
    return labels


def old_flatten64to1(semi, tensor=False):
    if tensor:
        is_batch = len(semi.shape) == 4
        if not is_batch:
            semi = semi.unsqueeze_(0)
        Hc, Wc = semi.shape[2], semi.shape[3]
        cell = 8
        semi.transpose_(1, 2)
        semi.transpose_(2, 3)
        semi = paddle.reshape(semi, shape=[-1, Hc, Wc, cell, cell])
        semi.transpose_(2, 3)
        semi = semi.contiguous()
        semi = paddle.reshape(semi, shape=[-1, 1, Hc * cell, Wc * cell])
        heatmap = semi
        if not is_batch:
            heatmap = heatmap.squeeze_(0)
    else:
        Hc, Wc = semi.shape[1], semi.shape[2]
        cell = 8
        semi = semi.transpose(1, 2, 0)
        heatmap = np.reshape(semi, [Hc, Wc, cell, cell])
        heatmap = np.transpose(heatmap, [0, 2, 1, 3])
        heatmap = np.reshape(heatmap, [Hc * cell, Wc * cell])
        heatmap = heatmap[np.newaxis, :, :]
    return heatmap


def flattenDetection(semi, tensor=False):
    batch = False
    if len(semi.shape) == 4:
        batch = True
        batch_size = semi.shape[0]
    if batch:
        dense = nn.functional.softmax(semi, axis=1)
        nodust = dense[:, :-1, :, :]
    else:
        dense = nn.functional.softmax(semi, axis=0)
        nodust = dense[:-1, :, :].unsqueeze(0)
    depth2space = DepthToSpace(8)
    heatmap = depth2space(nodust)
    heatmap = heatmap.squeeze(0) if not batch else heatmap
    return heatmap

"""
def sample_homo(image):
    from utils.homographies import sample_homography
    H = sample_homography(tf.shape(image)[:2])
    with tf.Session():
        H_ = H.eval()
    H_ = np.concatenate((H_, np.array([1])[:, np.newaxis]), axis=1)
    mat = np.reshape(H_, (3, 3))
    return mat
"""

import cv2


def getPtsFromHeatmap(heatmap, conf_thresh, nms_dist):
    border_remove = 4
    H, W = heatmap.shape[0], heatmap.shape[1]
    xs, ys = np.where(heatmap >= conf_thresh)
    sparsemap = heatmap >= conf_thresh
    if len(xs) == 0:
        return np.zeros((3, 0))
    pts = np.zeros((3, len(xs)))
    pts[0, :] = ys
    pts[1, :] = xs
    pts[2, :] = heatmap[xs, ys]
    pts, _ = nms_fast(pts, H, W, dist_thresh=nms_dist)
    inds = np.argsort(pts[2, :])
    pts = pts[:, inds[::-1]]
    bord = border_remove
    toremoveW = np.logical_or(pts[0, :] < bord, pts[0, :] >= W - bord)
    toremoveH = np.logical_or(pts[1, :] < bord, pts[1, :] >= H - bord)
    toremove = np.logical_or(toremoveW, toremoveH)
    pts = pts[:, ~toremove]
    return pts


def box_nms(prob, size, iou=0.1, min_prob=0.01, keep_top_k=0):
    from torchvision.ops import nms

    pts = paddle.to_tensor(paddle.nonzero(prob > min_prob), dtype=paddle.float32) # [N, 2]
    prob_nms = paddle.full_like(prob).requires_grad_(False)
    if pts.nelement() == 0:
        return prob_nms
    size = paddle.to_tensor(size / 2.0).cuda()
    boxes = paddle.concat([pts - size, pts + size], axis=1)
    scores = prob[paddle.to_tensor(pts[:, 0], dtype=paddle.int64), paddle.to_tensor(pts[:, 1], dtype=paddle.int64)]
    if keep_top_k != 0:
        indices = nms(boxes, scores, iou)
    else:
        raise NotImplementedError
    pts = paddle.index_select(pts, 0, indices)
    scores = paddle.index_select(scores, 0, indices)
    prob_nms[paddle.to_tensor(pts[:, 0], dtype=paddle.int64), paddle.to_tensor(pts[:, 1], dtype=paddle.int64)] = scores
    return prob_nms


def nms_fast(in_corners, H, W, dist_thresh):
    grid = np.zeros((H, W)).astype(int)
    inds = np.zeros((H, W)).astype(int)
    inds1 = np.argsort(-in_corners[2, :])
    corners = in_corners[:, inds1]
    rcorners = corners[:2, :].round().astype(int)
    if rcorners.shape[1] == 0:
        return np.zeros((3, 0)).astype(int), np.zeros(0).astype(int)
    if rcorners.shape[1] == 1:
        out = np.vstack((rcorners, in_corners[2])).reshape(3, 1)
        return out, np.zeros(1).astype(int)
    for i, rc in enumerate(rcorners.T):
        grid[rcorners[1, i], rcorners[0, i]] = 1
        inds[rcorners[1, i], rcorners[0, i]] = i
    pad = dist_thresh
    grid = np.pad(grid, ((pad, pad), (pad, pad)), mode='constant')
    count = 0
    for i, rc in enumerate(rcorners.T):
        pt = rc[0] + pad, rc[1] + pad
        if grid[pt[1], pt[0]] == 1:
            grid[pt[1] - pad:pt[1] + pad + 1, pt[0] - pad:pt[0] + pad + 1] = 0
            grid[pt[1], pt[0]] = -1
            count += 1
    keepy, keepx = np.where(grid == -1)
    keepy, keepx = keepy - pad, keepx - pad
    inds_keep = inds[keepy, keepx]
    out = corners[:, inds_keep]
    values = out[-1, :]
    inds2 = np.argsort(-values)
    out = out[:, inds2]
    out_inds = inds1[inds_keep[inds2]]
    return out, out_inds


def compute_valid_mask(image_shape, inv_homography, device='gpu', erosion_radius=0):
    if inv_homography.dim() == 2:
        inv_homography = paddle.reshape(inv_homography, shape=[-1, 3, 3])
    batch_size = inv_homography.shape[0]
    mask = paddle.ones([batch_size, 1, image_shape[0], image_shape[1]]
        ).requires_grad_(False)
    mask = inv_warp_image_batch(mask, inv_homography, device=device, mode='nearest')
    mask = paddle.reshape(mask, shape=[batch_size, image_shape[0], image_shape[1]])
    mask = mask.cpu().numpy()
    if erosion_radius > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                           (erosion_radius * 2,) * 2)
        for i in range(batch_size):
            mask[i, :, :] = cv2.erode(mask[i, :, :], kernel, iterations=1)
    return paddle.to_tensor(mask)


def normPts(pts, shape):
    pts = pts / shape * 2 - 1
    return pts


def denormPts(pts, shape):
    pts = (pts + 1) * shape / 2
    return pts


def descriptor_loss(descriptors, descriptors_warped, homographies, mask_valid=None, cell_size=8, lamda_d=250, device='gpu', descriptor_dist=4, **config):
    homographies = homographies
    from utils.utils import warp_points

    lamda_d = lamda_d
    margin_pos = 1
    margin_neg = 0.2
    batch_size, Hc, Wc = descriptors.shape[0], descriptors.shape[2
        ], descriptors.shape[3]
    H, W = Hc * cell_size, Wc * cell_size
    with paddle.no_grad():
        shape = paddle.to_tensor([H, W], dtype=paddle.float32)

        coor_cells = paddle.stack(paddle.meshgrid(paddle.arange(Hc), paddle.arange(Wc)), axis=2)
        coor_cells = paddle.to_tensor(coor_cells, dtype=paddle.float32)
        coor_cells = coor_cells * cell_size + cell_size // 2

        coor_cells = paddle.reshape(coor_cells, shape=[-1, 1, 1, Hc, Wc, 2])
        warped_coor_cells = normPts(paddle.reshape(coor_cells, shape=[-1, 2]), shape)
        warped_coor_cells = paddle.stack((warped_coor_cells[:, (1)], warped_coor_cells[:, (0)]), axis=1)
        warped_coor_cells = warp_points(warped_coor_cells, homographies, device)

        warped_coor_cells = paddle.stack((warped_coor_cells[:, :, (1)], warped_coor_cells[:, :, (0)]), axis=2)
        shape_cell = paddle.to_tensor([H // cell_size, W // cell_size], dtype=paddle.float32)

        warped_coor_cells = denormPts(warped_coor_cells, shape)
        warped_coor_cells = paddle.reshape(warped_coor_cells, shape=[-1, Hc, Wc, 1, 1, 2])

        cell_distances = coor_cells - warped_coor_cells
        cell_distances = paddle.norm(cell_distances, axis=-1)
        mask = cell_distances <= descriptor_dist

        mask = paddle.to_tensor(mask, dtype=paddle.float32)

    descriptors = descriptors.transpose(1, 2).transpose(2, 3)
    descriptors = paddle.reshape(descriptors, shape=[batch_size, Hc, Wc, 1, 1, -1])
    descriptors_warped = descriptors_warped.transpose(1, 2).transpose(2, 3)
    descriptors_warped = paddle.reshape(descriptors_warped, shape=[batch_size, 1, 1, Hc, Wc, -1])
    dot_product_desc = descriptors * descriptors_warped
    dot_product_desc = dot_product_desc.sum(dim=-1)
    positive_dist = paddle.max(margin_pos - dot_product_desc, paddle.to_tensor(0.0))
    negative_dist = paddle.max(dot_product_desc - margin_neg, paddle.to_tensor(0.0))

    if mask_valid is None:
        mask_valid = paddle.ones([batch_size, 1, Hc * cell_size, Wc * cell_size]).requires_grad_(False)
    mask_valid = paddle.reshape(mask_valid, shape=[batch_size, 1, 1, mask_valid.shape[2], mask_valid.shape[3]])

    loss_desc = lamda_d * mask * positive_dist + (1 - mask) * negative_dist
    loss_desc = loss_desc * mask_valid

    normalization = batch_size * (mask_valid.sum() + 1) * Hc * Wc
    pos_sum = (lamda_d * mask * positive_dist / normalization).sum()
    neg_sum = ((1 - mask) * negative_dist / normalization).sum()
    loss_desc = loss_desc.sum() / normalization

    return loss_desc, mask, pos_sum, neg_sum


def sumto2D(ndtensor):
    return ndtensor.sum(dim=1).sum(dim=1)


def mAP(pred_batch, labels_batch):
    pass


def precisionRecall_torch(pred, labels):
    offset = 10 ** -6
    assert pred.shape == labels.shape, 'Sizes of pred, labels should match when you get the precision/recall!'
    precision = paddle.sum(pred * labels) / (paddle.sum(pred) + offset)
    recall = paddle.sum(pred * labels) / (paddle.sum(labels) + offset)
    if precision.item() > 1.0:
        print(pred)
        print(labels)
        import scipy.io.savemat as savemat
        savemat('pre_recall.mat', {'pred': pred, 'labels': labels})
    assert precision.item() <= 1.0 and precision.item() >= 0.0
    return {'precision': precision, 'recall': recall}


def precisionRecall(pred, labels, thd=None):
    offset = 10 ** -6
    if thd is None:
        precision = np.sum(pred * labels) / (np.sum(pred) + offset)
        recall = np.sum(pred * labels) / (np.sum(labels) + offset)
    return {'precision': precision, 'recall': recall}


def getWriterPath(task='train', exper_name='', date=True):
    import datetime
    prefix = 'runs/'
    str_date_time = ''
    if exper_name != '':
        exper_name += '_'
    if date:
        str_date_time = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    return prefix + task + '/' + exper_name + str_date_time


def crop_or_pad_choice(in_num_points, out_num_points, shuffle=False):
    if shuffle:
        choice = np.random.permutation(in_num_points)
    else:
        choice = np.arange(in_num_points)
    assert out_num_points > 0, 'out_num_points = %d must be positive int!' % out_num_points
    if in_num_points >= out_num_points:
        choice = choice[:out_num_points]
    else:
        num_pad = out_num_points - in_num_points
        pad = np.random.choice(choice, num_pad, replace=True)
        choice = np.concatenate([choice, pad])
    return choice
