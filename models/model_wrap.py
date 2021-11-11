""" class to process superpoint net
# may be some duplication with model_wrap.py
# PointTracker is from Daniel's repo.
"""
import cv2
import numpy as np
from tqdm import tqdm

import paddle
import paddle.nn as nn


def labels2Dto3D(cell_size, labels):
    H, W = labels.shape[0], labels.shape[1]
    Hc, Wc = H // cell_size, W // cell_size

    labels = labels[:, np.newaxis, :, np.newaxis]
    labels = labels.reshape(Hc, cell_size, Wc, cell_size)
    labels = np.transpose(labels, [1, 3, 0, 2])
    labels = labels.reshape(1, cell_size ** 2, Hc, Wc)
    labels = labels.squeeze()

    dustbin = labels.sum(axis=0)
    dustbin = 1 - dustbin
    dustbin[dustbin < 0] = 0

    labels = np.concatenate((labels, dustbin[np.newaxis, :, :]), axis=0)
    return labels


def toNumpy(tensor):
    return tensor.detach().cpu().numpy()


class SuperPointFrontend_torch(object):

    def __init__(self, config, weights_path, nms_dist, conf_thresh,
                 nn_thresh, cuda=False, trained=False, device='gpu', grad=False, load=True):
        self.config = config
        self.name = 'SuperPoint'
        self.cuda = cuda
        self.nms_dist = nms_dist
        self.conf_thresh = conf_thresh
        self.nn_thresh = nn_thresh
        self.cell = 8
        self.border_remove = 4
        self.sparsemap = None
        self.heatmap = None
        self.pts = None
        self.pts_subpixel = None
        self.patches = None
        self.device = device
        self.subpixel = False
        if self.config['model']['subpixel']['enable']:
            self.subpixel = True
        if load:
            self.loadModel(weights_path)

    def loadModel(self, weights_path):
        if weights_path[-4:] == '.tar':
            trained = True

        if trained:
            model = self.config['model']['name']
            params = self.config['model']['params']
            print('model: ', model)

            from utils.loader import modelLoader

            self.net = modelLoader(model=model, **params)
            checkpoint = paddle.load(weights_path)
            self.net.load_dict(checkpoint['model_state_dict'])
        
        else:
            from models.SuperPointNet_pretrained import SuperPointNet

            self.net = SuperPointNet()
            self.net.load_dict(paddle.load(weights_path))

    def net_parallel(self):
        print("=== Let's use", paddle.get_device(), 'GPUs!')
        self.net = paddle.DataParallel(self.net)

    def nms_fast(self, in_corners, H, W, dist_thresh):
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
                grid[pt[1] - pad:pt[1] + pad + 1, pt[0] - pad:pt[0] + pad + 1
                    ] = 0
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

    def getSparsemap(self):
        return self.sparsemap

    @property
    def points(self):
        print('get pts')
        return self.pts

    @property
    def heatmap(self):
        return self._heatmap

    @heatmap.setter
    def heatmap(self, heatmap):
        self._heatmap = heatmap

    def soft_argmax_points(self, pts, patch_size=5):
        from utils.utils import toNumpy
        from utils.losses import extract_patch_from_points
        from utils.losses import soft_argmax_2d
        from utils.losses import norm_patches

        pts = pts[0].transpose().copy()
        patches = extract_patch_from_points(self.heatmap, pts, patch_size=patch_size)
        
        import paddle
        
        patches = np.stack(patches)
        patches_torch = paddle.to_tensor(patches, dtype=paddle.float32
            ).unsqueeze(0)
        patches_torch = norm_patches(patches_torch)
        from utils.losses import do_log
        patches_torch = do_log(patches_torch)
        dxdy = soft_argmax_2d(patches_torch, normalized_coordinates=False)
        points = pts
        points[:, :2] = points[:, :2] + dxdy.numpy().squeeze() - patch_size // 2
        self.patches = patches_torch.numpy().squeeze()
        self.pts_subpixel = [points.transpose().copy()]
        return self.pts_subpixel.copy()

    def get_image_patches(self, pts, image, patch_size=5):
        from utils.losses import extract_patch_from_points
        pts = pts[0].transpose().copy()
        patches = extract_patch_from_points(image, pts, patch_size=patch_size)
        patches = np.stack(patches)
        return patches

    def getPtsFromHeatmap(self, heatmap):
        heatmap = heatmap.squeeze()
        H, W = heatmap.shape[0], heatmap.shape[1]
        xs, ys = np.where(heatmap >= self.conf_thresh)
        self.sparsemap = heatmap >= self.conf_thresh
        if len(xs) == 0:
            return np.zeros((3, 0))
        pts = np.zeros((3, len(xs)))
        pts[0, :] = ys
        pts[1, :] = xs
        pts[2, :] = heatmap[xs, ys]
        pts, _ = self.nms_fast(pts, H, W, dist_thresh=self.nms_dist)
        inds = np.argsort(pts[2, :])
        pts = pts[:, inds[::-1]]
        bord = self.border_remove
        toremoveW = np.logical_or(pts[0, :] < bord, pts[0, :] >= W - bord)
        toremoveH = np.logical_or(pts[1, :] < bord, pts[1, :] >= H - bord)
        toremove = np.logical_or(toremoveW, toremoveH)
        pts = pts[:, ~toremove]
        return pts

    def sample_desc_from_points(self, coarse_desc, pts):
        H, W = coarse_desc.shape[2] * self.cell, coarse_desc.shape[3
            ] * self.cell
        D = coarse_desc.shape[1]
        if pts.shape[1] == 0:
            desc = np.zeros((D, 0))
        else:
            samp_pts = paddle.to_tensor(pts[:2, :].copy())
            samp_pts[0, :] = samp_pts[0, :] / (float(W) / 2.0) - 1.0
            samp_pts[1, :] = samp_pts[1, :] / (float(H) / 2.0) - 1.0
            samp_pts = paddle.transpose(samp_pts, perm=[0, 1])
                #.contiguous()
            samp_pts = paddle.reshape(samp_pts, shape=[1, 1, -1, 2])
            samp_pts = paddle.to_tensor(samp_pts, dtype=paddle.float32)

            desc = paddle.nn.functional.grid_sample(coarse_desc, samp_pts, align_corners=True)
            desc = desc.numpy().reshape(D, -1)
            desc /= np.linalg.norm(desc, axis=0)[np.newaxis, :]
        return desc

    def subpixel_predict(self, pred_res, points, verbose=False):
        D = points.shape[0]
        if points.shape[1] == 0:
            pts_subpixel = np.zeros((D, 0))
        else:
            points_res = pred_res[:, points[(1), :].astype(int), points[(0),
                :].astype(int)]
            pts_subpixel = points.copy()
            if verbose:
                print('before: ', pts_subpixel[:, :5])
            pts_subpixel[:2, :] += points_res
            if verbose:
                print('after: ', pts_subpixel[:, :5])
        return pts_subpixel
        pass

    def run(self, inp, onlyHeatmap=False, train=True):
        batch_size, H, W = inp.shape[0], inp.shape[2], inp.shape[3]
        if train:
            outs = self.net.forward(inp)
            semi, coarse_desc = outs['semi'], outs['desc']
        else:
            with paddle.no_grad():
                outs = self.net.forward(inp)
                semi, coarse_desc = outs['semi'], outs['desc']

        from utils.utils import labels2Dto3D
        from utils.utils import flattenDetection
        from utils.d2s import DepthToSpace

        heatmap = flattenDetection(semi, tensor=True)
        self.heatmap = heatmap
        if onlyHeatmap:
            return heatmap
        pts = [self.getPtsFromHeatmap(heatmap[i, :, :, :].cpu().detach().
            numpy()) for i in range(batch_size)]
        self.pts = pts
        if self.subpixel:
            labels_res = outs[2]
            self.pts_subpixel = [self.subpixel_predict(toNumpy(labels_res[i,
                ...]), pts[i]) for i in range(batch_size)]

        dense_desc = nn.functional.interpolate(coarse_desc, scale_factor=(
            self.cell, self.cell), mode='bilinear')

        def norm_desc(desc):
            dn = paddle.norm(desc, p=2, axis=1) # Compute the norm.
            desc = desc.div(paddle.unsqueeze(dn, 1))
            return desc

        dense_desc = norm_desc(dense_desc)
        dense_desc_cpu = dense_desc.cpu().detach().numpy()
        pts_desc = [dense_desc_cpu[i, :, pts[i][1, :].astype(int), pts[i][0, :].astype(int)].transpose() for i in range(len(pts))]

        if self.subpixel:
            return self.pts_subpixel, pts_desc, dense_desc, heatmap
        return pts, pts_desc, dense_desc, heatmap


class PointTracker(object):

    def __init__(self, max_length=2, nn_thresh=0.7):
        if max_length < 2:
            raise ValueError('max_length must be greater than or equal to 2.')
        self.maxl = max_length
        self.nn_thresh = nn_thresh
        self.all_pts = []
        for n in range(self.maxl):
            self.all_pts.append(np.zeros((2, 0)))

        self.last_desc = None
        self.tracks = np.zeros((0, self.maxl + 2))
        self.track_count = 0
        self.max_score = 9999
        self.matches = None
        self.last_pts = None
        self.mscores = None

    def nn_match_two_way(self, desc1, desc2, nn_thresh):
        assert desc1.shape[0] == desc2.shape[0]
        if desc1.shape[1] == 0 or desc2.shape[1] == 0:
            return np.zeros((3, 0))

        if nn_thresh < 0.0:
            raise ValueError("'nn_thresh' should be non-negative")

        dmat = np.dot(desc1.T, desc2)
        dmat = np.sqrt(2 - 2 * np.clip(dmat, -1, 1))

        idx = np.argmin(dmat, axis=1)
        scores = dmat[np.arange(dmat.shape[0]), idx]
        keep = scores < nn_thresh
        idx2 = np.argmin(dmat, axis=0)

        keep_bi = np.arange(len(idx)) == idx2[idx]
        keep = np.logical_and(keep, keep_bi)

        idx = idx[keep]
        scores = scores[keep]

        m_idx1 = np.arange(desc1.shape[1])[keep]
        m_idx2 = idx

        matches = np.zeros((3, int(keep.sum())))
        matches[0, :] = m_idx1
        matches[1, :] = m_idx2
        matches[2, :] = scores

        self.mscores = matches
        return matches

    def get_offsets(self):
        offsets = []
        offsets.append(0)

        for i in range(len(self.all_pts) - 1):
            offsets.append(self.all_pts[i].shape[1])

        offsets = np.array(offsets)
        offsets = np.cumsum(offsets)
        return offsets

    def get_matches(self):
        return self.matches

    def get_mscores(self):
        return self.mscores

    def clear_desc(self):
        self.last_desc = None

    def update(self, pts, desc):
        if pts is None or desc is None:
            print('PointTracker: Warning, no points were added to tracker.')
            return

        assert pts.shape[1] == desc.shape[1]
        if self.last_desc is None:
            self.last_desc = np.zeros((desc.shape[0], 0))

        remove_size = self.all_pts[0].shape[1]

        self.all_pts.pop(0)
        self.all_pts.append(pts)
        self.tracks = np.delete(self.tracks, 2, axis=1)

        for i in range(2, self.tracks.shape[1]):
            self.tracks[:, i] -= remove_size

        self.tracks[:, 2:][self.tracks[:, 2:] < -1] = -1
        offsets = self.get_offsets()
        self.tracks = np.hstack((self.tracks, -1 * np.ones((self.tracks.shape[0], 1))))

        matched = np.zeros(pts.shape[1]).astype(bool)
        matches = self.nn_match_two_way(self.last_desc, desc, self.nn_thresh)
        self.matches = matches

        pts_id = pts[:2, :]

        if self.last_pts is not None:
            id1, id2 = self.last_pts[:, matches[(0), :].astype(int)], pts_id[:, matches[(1), :].astype(int)]
            self.matches = np.concatenate((id1, id2), axis=0)

        for match in matches.T:
            id1 = int(match[0]) + offsets[-2]
            id2 = int(match[1]) + offsets[-1]

            found = np.argwhere(self.tracks[:, -2] == id1)

            if found.shape[0] > 0:
                matched[int(match[1])] = True
                row = int(found)
                self.tracks[row, -1] = id2

                if self.tracks[row, 1] == self.max_score:
                    self.tracks[row, 1] = match[2]
                else:
                    track_len = (self.tracks[row, 2:] != -1).sum() - 1.0
                    frac = 1.0 / float(track_len)
                    self.tracks[row, 1] = (1.0 - frac) * self.tracks[row, 1] + frac * match[2]

        new_ids = np.arange(pts.shape[1]) + offsets[-1]
        new_ids = new_ids[~matched]

        new_tracks = -1 * np.ones((new_ids.shape[0], self.maxl + 2))
        new_tracks[:, -1] = new_ids

        new_num = new_ids.shape[0]

        new_trackids = self.track_count + np.arange(new_num)
        new_tracks[:, 0] = new_trackids
        new_tracks[:, 1] = self.max_score * np.ones(new_ids.shape[0])

        self.tracks = np.vstack((self.tracks, new_tracks))
        self.track_count += new_num

        keep_rows = np.any(self.tracks[:, 2:] >= 0, axis=1)

        self.tracks = self.tracks[keep_rows, :]

        self.last_desc = desc.copy()
        self.last_pts = pts[:2, :].copy()
        return

    def get_tracks(self, min_length):
        if min_length < 1:
            raise ValueError("'min_length' too small.")

        valid = np.ones(self.tracks.shape[0]).astype(bool)

        good_len = np.sum(self.tracks[:, 2:] != -1, axis=1) >= min_length
        not_headless = self.tracks[:, -1] != -1

        keepers = np.logical_and.reduce((valid, good_len, not_headless))
        returned_tracks = self.tracks[keepers, :].copy()

        return returned_tracks

    def draw_tracks(self, out, tracks):
        pts_mem = self.all_pts
        N = len(pts_mem)
        offsets = self.get_offsets()
        stroke = 1

        for track in tracks:
            clr = myjet[int(np.clip(np.floor(track[1] * 10), 0, 9)), :] * 255

            for i in range(N - 1):
                if track[i + 2] == -1 or track[i + 3] == -1:
                    continue
                offset1 = offsets[i]
                offset2 = offsets[i + 1]

                idx1 = int(track[i + 2] - offset1)
                idx2 = int(track[i + 3] - offset2)

                pt1 = pts_mem[i][:2, idx1]
                pt2 = pts_mem[i + 1][:2, idx2]

                p1 = int(round(pt1[0])), int(round(pt1[1]))
                p2 = int(round(pt2[0])), int(round(pt2[1]))

                cv2.line(out, p1, p2, clr, thickness=stroke, lineType=16)

                if i == N - 2:
                    clr2 = 255, 0, 0
                    cv2.circle(out, p2, stroke, clr2, -1, lineType=16)
