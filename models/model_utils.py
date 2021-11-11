""" class to process superpoint net
# may be some duplication with model_wrap.py
"""
import paddle
import numpy as np

from utils.t2p import SpatialSoftArgmax2d


class SuperPointNet_process(object):

    def __init__(self, **config):
        self.out_num_points = config.get('out_num_points', 500)
        self.patch_size = config.get('patch_size', 5)
        self.device = config.get('device', 'cuda:0')
        self.nms_dist = config.get('nms_dist', 4)
        self.conf_thresh = config.get('conf_thresh', 0.015)
        self.heatmap = None
        self.heatmap_nms_batch = None
        pass

    def pred_soft_argmax(self, labels_2D, heatmap):

        patch_size = self.patch_size
        device = self.device
        from utils.losses import norm_patches
        outs = {}
        from utils.losses import extract_patches
        from utils.losses import soft_argmax_2d
        label_idx = labels_2D[...].nonzero()
        patches = extract_patches(label_idx, heatmap,
            patch_size=patch_size)
        from utils.losses import do_log
        patches_log = do_log(patches)
        dxdy = soft_argmax_2d(patches_log, normalized_coordinates=False)
        dxdy = dxdy.squeeze(1)
        dxdy = dxdy - patch_size // 2
        outs['pred'] = dxdy
        outs['patches'] = patches
        return outs

    @staticmethod
    def sample_desc_from_points(coarse_desc, pts, cell_size=8):

        samp_pts = pts.transpose(0, 1)
        H, W = coarse_desc.shape[2] * cell_size, coarse_desc.shape[3
            ] * cell_size
        D = coarse_desc.shape[1]
        if pts.shape[1] == 0:
            desc = paddle.ones((1, 1, D)).requires_grad_(False)
        else:
            samp_pts[0, :] = samp_pts[0, :] / (float(W) / 2.0) - 1.0
            samp_pts[1, :] = samp_pts[1, :] / (float(H) / 2.0) - 1.0
            samp_pts = samp_pts.transpose(0, 1).contiguous()
            samp_pts = paddle.reshape(samp_pts, shape=[1, 1, -1, 2])
            samp_pts = paddle.to_tensor(samp_pts, dtype=paddle.float32)
            desc = paddle.nn.functional.grid_sample(coarse_desc, samp_pts, align_corners=True) # tensor [batch_size(1), D, 1, N]
            desc = desc.squeeze().transpose(0, 1).unsqueeze(0)
        return desc

    @staticmethod
    def ext_from_points(labels_res, points):

        labels_res = labels_res.transpose(1, 2).transpose(2, 3).unsqueeze(1)
        points_res = labels_res[points[:, (0)], points[:, (1)], points[:, (
            2)], points[:, (3)], :]
        return points_res

    @staticmethod
    def soft_argmax_2d(patches):

        m = SpatialSoftArgmax2d()
        coords = m(patches)
        return coords

    def heatmap_to_nms(self, heatmap, tensor=False, boxnms=False):

        to_floatTensor = lambda x: paddle.to_tensor(x, dtype=paddle.float32)
        from utils.var_dim import toNumpy
        heatmap_np = toNumpy(heatmap)
        if boxnms:
            from utils.utils import box_nms
            heatmap_nms_batch = [box_nms(h.detach().squeeze(), self.
                nms_dist, min_prob=self.conf_thresh) for h in heatmap]
            heatmap_nms_batch = paddle.stack(heatmap_nms_batch, axis=0
                ).unsqueeze(1)
        else:
            heatmap_nms_batch = [self.heatmap_nms(h, self.nms_dist, self.
                conf_thresh) for h in heatmap_np]
            heatmap_nms_batch = np.stack(heatmap_nms_batch, axis=0)
            heatmap_nms_batch = heatmap_nms_batch[:, np.newaxis, ...]
            if tensor:
                heatmap_nms_batch = to_floatTensor(heatmap_nms_batch)
        self.heatmap = heatmap
        self.heatmap_nms_batch = heatmap_nms_batch
        return heatmap_nms_batch
        pass

    @staticmethod
    def heatmap_nms(heatmap, nms_dist=4, conf_thresh=0.015):

        heatmap = heatmap.squeeze()
        boxnms = False
        from utils.utils import getPtsFromHeatmap
        pts_nms = getPtsFromHeatmap(heatmap, conf_thresh, nms_dist)
        semi_thd_nms_sample = np.zeros_like(heatmap)
        semi_thd_nms_sample[pts_nms[1, :].astype(np.int), pts_nms[0, :].
            astype(np.int)] = 1
        return semi_thd_nms_sample

    def batch_extract_features(self, desc, heatmap_nms_batch, residual):
        from utils.utils import crop_or_pad_choice
        batch_size = heatmap_nms_batch.shape[0]
        pts_int, pts_offset, pts_desc = [], [], []
        pts_idx = heatmap_nms_batch[...].nonzero()
        for i in range(batch_size):
            mask_b = pts_idx[:, 0] == i
            pts_int_b = paddle.to_tensor(pts_idx[mask_b][:, 2:], dtype=paddle.float32)
            pts_int_b = pts_int_b[:, [1, 0]]
            res_b = residual[mask_b]
            pts_b = pts_int_b + res_b
            pts_desc_b = self.sample_desc_from_points(desc[i].unsqueeze(0),
                pts_b).squeeze(0)
            from utils.utils import crop_or_pad_choice
            choice = crop_or_pad_choice(pts_int_b.shape[0], out_num_points=\
                self.out_num_points, shuffle=True)
            choice = paddle.to_tensor(choice)
            pts_int.append(pts_int_b[choice])
            pts_offset.append(res_b[choice])
            pts_desc.append(pts_desc_b[choice])
        pts_int = paddle.stack(pts_int, axis=0)
        pts_offset = paddle.stack(pts_offset, axis=0)
        pts_desc = paddle.stack(pts_desc, axis=0)
        return {'pts_int': pts_int, 'pts_offset': pts_offset, 'pts_desc':
            pts_desc}
