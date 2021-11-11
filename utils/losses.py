"""losses
# losses for heatmap residule
# use it if you're computing residual loss. 
# current disable residual loss
"""
import paddle
import utils.t2p


def print_var(points):
    print('points: ', points.shape)
    print('points: ', points)
    pass


def pts_to_bbox(points, patch_size):
    shift_l = (patch_size + 1) / 2
    shift_r = patch_size - shift_l
    pts_l = points - shift_l
    pts_r = points + shift_r + 1
    bbox = paddle.stack((pts_l[:, (1)], pts_l[:, (0)], pts_r[:, (1)], pts_r[:, (0)]), axis=1)
    return bbox
    pass

"""
def _roi_pool(pred_heatmap, rois, patch_size=8):
    from torchvision.ops import roi_pool
    patches = roi_pool(pred_heatmap, paddle.to_tensor(rois, dtype=paddle.float32), (patch_size, patch_size), spatial_scale=1.0)
    return patches
    pass
"""

def norm_patches(patches):
    patch_size = patches.shape[-1]
    patches = paddle.reshape(patches, shape=[-1, 1, patch_size * patch_size])
    d = paddle.sum(patches, axis=-1).unsqueeze(-1) + 1e-06
    patches = patches / d
    patches = paddle.reshape(patches, shape=[-1, 1, patch_size, patch_size])

    return patches


def extract_patch_from_points(heatmap, points, patch_size=5):
    import numpy as np
    from utils.utils import toNumpy

    if isinstance(heatmap, paddle.Tensor):
        heatmap = toNumpy(heatmap)
    heatmap = heatmap.squeeze()

    pad_size = int(patch_size / 2)
    heatmap = np.pad(heatmap, pad_size, 'constant')
    patches = []
    ext = lambda img, pnt, wid: img[pnt[1]:pnt[1] + wid, pnt[0]:pnt[0] + wid]
    print('heatmap: ', heatmap.shape)
    for i in range(points.shape[0]):
        patch = ext(heatmap, points[i, :].astype(int), patch_size)
        patches.append(patch)

    return patches


def extract_patches(label_idx, image, patch_size=7):
    rois = paddle.to_tensor(pts_to_bbox(label_idx[:, 2:], patch_size), dtype=paddle.int64)

    rois = paddle.concat((label_idx[:, :1], rois), axis=1)

    patches = _roi_pool(image, rois, patch_size=patch_size)
    return patches


def points_to_4d(points):
    num_of_points = points.shape[0]
    cols = paddle.to_tensor(paddle.zeros([num_of_points, 1]).requires_grad_(False), dtype=float32)
    points = paddle.concat((cols, cols, paddle.to_tensor(points, dtype=paddle.float32)), axis=1)
    return points


def soft_argmax_2d(patches, normalized_coordinates=True):
    m = utils.t2p.SpatialSoftArgmax2d(normalized_coordinates=normalized_coordinates)
    coords = m(patches)
    return coords


def do_log(patches):
    patches[patches < 0] = 1e-06
    patches_log = paddle.log(patches)
    return patches_log


def subpixel_loss(labels_2D, labels_res, pred_heatmap, patch_size=7):

    def _soft_argmax(patches):
        from models.SubpixelNet import SubpixelNet as subpixNet
        dxdy = subpixNet.soft_argmax_2d(patches)
        dxdy = dxdy.squeeze(1)
        return dxdy

    points = labels_2D[...].nonzero()
    num_points = points.shape[0]
    if num_points == 0:
        return 0

    labels_res = labels_res.transpose(1, 2).transpose(2, 3).unsqueeze(1)
    rois = pts_to_bbox(points[:, 2:], patch_size)

    rois = paddle.concat((points[:, :1], rois), axis=1)
    points_res = labels_res[points[:, (0)], points[:, (1)], points[:, (2)], points[:, (3)], :]

    patches = _roi_pool(pred_heatmap, rois, patch_size=patch_size)

    dxdy = _soft_argmax(patches)

    loss = points_res - dxdy
    loss = paddle.norm(loss, p=2, axis=-1)
    loss = loss.sum() / num_points
    return loss


def subpixel_loss_no_argmax(labels_2D, labels_res, pred_heatmap, **options):
    points = labels_2D[...].nonzero()
    num_points = points.shape[0]
    if num_points == 0:
        return 0

    def residual_from_points(labels_res, points):
        labels_res = labels_res.transpose(1, 2).transpose(2, 3).unsqueeze(1)
        points_res = labels_res[points[:, (0)], points[:, (1)], points[:, 2], points[:, (3)], :]
        return points_res

    points_res = residual_from_points(labels_res, points)

    pred_res = residual_from_points(pred_heatmap, points)

    loss = points_res - pred_res
    loss = paddle.norm(loss, p=2, axis=-1).mean()

    return loss
    pass


if __name__ == '__main__':
    pass
