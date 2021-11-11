import numpy as np
from os import path as osp
from glob import glob

from settings import EXPER_PATH


def get_paths(exper_name):
    return glob(osp.join(EXPER_PATH, 'outputs/{}/*.npz'.format(exper_name)))


def compute_tp_fp(data, remove_zero=0.0001, distance_thresh=2, simplified=False):
    gt = np.where(data['keypoint_map'])
    gt = np.stack([gt[0], gt[1]], axis=-1)
    n_gt = len(gt)
    prob = data['prob_nms'] if 'prob_nms' in data.files else data['prob']

    mask = np.where(prob > remove_zero)
    prob = prob[mask]
    pred = np.array(mask).T

    sort_idx = np.argsort(prob)[::-1]
    prob = prob[sort_idx]
    pred = pred[sort_idx]

    diff = np.expand_dims(pred, axis=1) - np.expand_dims(gt, axis=0)
    dist = np.linalg.norm(diff, axis=-1)
    matches = np.less_equal(dist, distance_thresh)

    tp = []
    matched = np.zeros(len(gt))
    for m in matches:
        correct = np.any(m)
        if correct:
            gt_idx = np.argmax(m)
            tp.append(not matched[gt_idx])
            matched[gt_idx] = 1
        else:
            tp.append(False)
    tp = np.array(tp, bool)
    if simplified:
        tp = np.any(matches, axis=1)
        n_gt = np.sum(np.minimum(np.sum(matches, axis=0), 1))
    fp = np.logical_not(tp)
    return tp, fp, prob, n_gt


def div0(a, b):
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.true_divide(a, b)
        idx = ~np.isfinite(c)
        c[idx] = np.where(a[idx] == 0, 1, 0)
    return c


def compute_pr(exper_name, **kwargs):
    paths = get_paths(exper_name)
    tp, fp, prob, n_gt = [], [], [], 0
    for path in paths:
        t, f, p, n = compute_tp_fp(np.load(path), **kwargs)
        tp.append(t)
        fp.append(f)
        prob.append(p)
        n_gt += n
    tp = np.concatenate(tp)
    fp = np.concatenate(fp)
    prob = np.concatenate(prob)

    sort_idx = np.argsort(prob)[::-1]
    tp = tp[sort_idx]
    fp = fp[sort_idx]
    prob = prob[sort_idx]

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recall = div0(tp_cum, n_gt)
    precision = div0(tp_cum, tp_cum + fp_cum)
    recall = np.concatenate([[0], recall, [1]])
    precision = np.concatenate([[0], precision, [0]])
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    return precision, recall, prob


def compute_mAP(precision, recall):
    return np.sum(precision[1:] * (recall[1:] - recall[:-1]))


def compute_loc_error(exper_name, prob_thresh=0.5, distance_thresh=2):

    def loc_error_per_image(data):
        gt = np.where(data['keypoint_map'])
        gt = np.stack([gt[0], gt[1]], axis=-1)
        prob = data['prob']

        mask = np.where(prob > prob_thresh)
        pred = np.array(mask).T
        prob = prob[mask]

        if not len(gt) or not len(pred):
            return []

        diff = np.expand_dims(pred, axis=1) - np.expand_dims(gt, axis=0)
        dist = np.linalg.norm(diff, axis=-1)
        dist = np.min(dist, axis=1)
        correct_dist = dist[np.less_equal(dist, distance_thresh)]
        return correct_dist
    paths = get_paths(exper_name)
    error = []
    for path in paths:
        error.append(loc_error_per_image(np.load(path)))
    return np.mean(np.concatenate(error))


def warp_keypoints(keypoints, H):
    num_points = keypoints.shape[0]
    homogeneous_points = np.concatenate([keypoints, np.ones((num_points, 1))], axis=1)
    warped_points = np.dot(homogeneous_points, np.transpose(H))
    return warped_points[:, :2] / warped_points[:, 2:]


def compute_repeatability(data, keep_k_points=300, distance_thresh=3, verbose=False):

    def filter_keypoints(points, shape):
        mask = (points[:, 0] >= 0) & (points[:, 0] < shape[1]) &\
               (points[:, 1] >= 0) & (points[:, 1] < shape[0])
        return points[mask, :]

    def keep_true_keypoints(points, H, shape):
        warped_points = warp_keypoints(points[:, [0, 1]], H)
        mask = (warped_points[:, 0] >= 0) & (warped_points[:, 0] < shape[1]) &\
               (warped_points[:, 1] >= 0) & (warped_points[:, 1] < shape[0])
        return points[mask, :]

    def select_k_best(points, k):
        sorted_prob = points
        if points.shape[1] > 2:
            sorted_prob = points[points[:, (2)].argsort(), :2]
            start = min(k, points.shape[0])
            sorted_prob = sorted_prob[-start:, :]
        return sorted_prob

    localization_err = -1
    repeatability = []
    N1s = []
    N2s = []

    shape = data['image'].shape
    H = data['homography']

    keypoints = data['prob']

    warped_keypoints = data['warped_prob']

    warped_keypoints = keep_true_keypoints(warped_keypoints,
                                           np.linalg.inv(H),
                                           data['image'].shape)

    true_warped_keypoints = keypoints

    true_warped_keypoints[:, :2] = warp_keypoints(keypoints[:, :2], H)

    true_warped_keypoints = filter_keypoints(true_warped_keypoints, shape)

    warped_keypoints = select_k_best(warped_keypoints, keep_k_points)
    true_warped_keypoints = select_k_best(true_warped_keypoints, keep_k_points)

    N1 = true_warped_keypoints.shape[0]
    print('true_warped_keypoints: ', true_warped_keypoints[:2, :])
    N2 = warped_keypoints.shape[0]
    print('warped_keypoints: ', warped_keypoints[:2, :])
    N1s.append(N1)
    N2s.append(N2)
    true_warped_keypoints = np.expand_dims(true_warped_keypoints, 1)
    warped_keypoints = np.expand_dims(warped_keypoints, 0)

    norm = np.linalg.norm(true_warped_keypoints - warped_keypoints,
                          ord=None, axis=2)
    count1 = 0
    count2 = 0
    local_err1, local_err2 = None, None
    if N2 != 0:
        min1 = np.min(norm, axis=1)
        count1 = np.sum(min1 <= distance_thresh)

        local_err1 = min1[min1 <= distance_thresh]
    if N1 != 0:
        min2 = np.min(norm, axis=0)
        count2 = np.sum(min2 <= distance_thresh)
        local_err2 = min2[min2 <= distance_thresh]

    if N1 + N2 > 0:
        repeatability = (count1 + count2) / (N1 + N2)
    if count1 + count2 > 0:
        localization_err = 0
        if local_err1 is not None:
            localization_err += local_err1.sum() / (count1 + count2)
        if local_err2 is not None:
            localization_err += local_err2.sum() / (count1 + count2)
    else:
        repeatability = 0
    if verbose:
        print('Average number of points in the first image: ' + str(np.mean(N1s)))
        print('Average number of points in the second image: ' + str(np.mean(N2s)))

    return repeatability, localization_err
