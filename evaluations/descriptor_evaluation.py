"""Script for descriptor evaluation
"""
import numpy as np
import cv2
from os import path as osp
from glob import glob

from settings import EXPER_PATH


def get_paths(exper_name):
    return glob(osp.join(EXPER_PATH, 'outputs/{}/*.npz'.format(exper_name)))


def keep_shared_points(keypoint_map, H, keep_k_points=1000):

    def select_k_best(points, k):
        sorted_prob = points[points[:, (2)].argsort(), :2]
        start = min(k, points.shape[0])
        return sorted_prob[-start:, :]

    def warp_keypoints(keypoints, H):
        num_points = keypoints.shape[0]
        homogeneous_points = np.concatenate([keypoints, np.ones((num_points,1))], axis=1)
        warped_points = np.dot(homogeneous_points, np.transpose(H))
        return warped_points[:, :2] / warped_points[:, 2:]

    def keep_true_keypoints(points, H, shape):
        warped_points = warp_keypoints(points[:, [1, 0]], H)
        warped_points[:, [0, 1]] = warped_points[:, [1, 0]]
        mask = (warped_points[:, 0] >= 0) & (warped_points[:, 0] < shape[0]
            ) & (warped_points[:, 1] >= 0) & (warped_points[:, 1] < shape[1])
        return points[mask, :]

    keypoints = np.where(keypoint_map > 0)
    prob = keypoint_map[keypoints[0], keypoints[1]]
    keypoints = np.stack([keypoints[0], keypoints[1], prob], axis=-1)
    keypoints = keep_true_keypoints(keypoints, H, keypoint_map.shape)
    keypoints = select_k_best(keypoints, keep_k_points)

    return keypoints.astype(int)


def compute_homography(data, keep_k_points=1000, correctness_thresh=1, orb=False, shape=(240, 320)):

    print('shape: ', shape)
    real_H = data['homography']

    keypoints = data['prob'][:, [1, 0]]

    warped_keypoints = data['warped_prob'][:, [1, 0]]

    desc = data['desc']
    warped_desc = data['warped_desc']

    if orb:
        desc = desc.astype(np.uint8)
        warped_desc = warped_desc.astype(np.uint8)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    else:
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    print('desc: ', desc.shape)
    print('w desc: ', warped_desc.shape)
    cv2_matches = bf.match(desc, warped_desc)
    matches_idx = np.array([m.queryIdx for m in cv2_matches])
    m_keypoints = keypoints[matches_idx, :]
    matches_idx = np.array([m.trainIdx for m in cv2_matches])
    m_dist = np.array([m.distance for m in cv2_matches])
    m_warped_keypoints = warped_keypoints[matches_idx, :]
    matches = np.hstack((m_keypoints[:, [1, 0]], m_warped_keypoints[:, [1, 0]]))
    print(f'matches: {matches.shape}')

    H, inliers = cv2.findHomography(m_keypoints[:, [1, 0]],
                                    m_warped_keypoints[:, [1, 0]],
                                    cv2.RANSAC)

    inliers = inliers.flatten()

    if H is None:
        correctness = 0
        H = np.identity(3)
        print('no valid estimation')
    else:
        corners = np.array([[0, 0, 1],
                            [0, shape[0] - 1, 1],
                            [shape[1] - 1, 0, 1],
                            [shape[1] - 1, shape[0] - 1, 1]])
        print('corner: ', corners)

        real_warped_corners = np.dot(corners, np.transpose(real_H))
        real_warped_corners = real_warped_corners[:, :2] / real_warped_corners[:, 2:]
        print('real_warped_corners: ', real_warped_corners)

        warped_corners = np.dot(corners, np.transpose(H))
        warped_corners = warped_corners[:, :2] / warped_corners[:, 2:]
        print('warped_corners: ', warped_corners)

        mean_dist = np.mean(np.linalg.norm(real_warped_corners - warped_corners, axis=1))

        correctness = mean_dist <= correctness_thresh

    return {'correctness': correctness,
            'keypoints1': keypoints,
            'keypoints2': warped_keypoints,
            'matches': matches,
            'cv2_matches': cv2_matches,
            'mscores': m_dist / m_dist.max(),
            'inliers': inliers,
            'homography': H,
            'mean_dist': mean_dist}


def homography_estimation(exper_name, keep_k_points=1000, correctness_thresh=1, orb=False):
    paths = get_paths(exper_name)
    correctness = []
    for path in paths:
        data = np.load(path)
        estimates = compute_homography(data, keep_k_points,
                                       correctness_thresh, orb)
        correctness.append(estimates['correctness'])
    return np.mean(correctness)


def get_homography_matches(exper_name, keep_k_points=1000, correctness_thresh=1, num_images=1, orb=False):

    paths = get_paths(exper_name)
    outputs = []
    for path in paths[:num_images]:
        data = np.load(path)
        output = compute_homography(data, keep_k_points, correctness_thresh, orb)
        output['image1'] = data['image']
        output['image2'] = data['warped_image']
        outputs.append(output)
    return outputs
