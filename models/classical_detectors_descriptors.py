
import numpy as np
import cv2
from matplotlib import pyplot as plt


def classical_detector_descriptor(im, **config):
    im = np.uint8(im)
    if config['method'] == 'sift':
        sift = cv2.xfeatures2d.SIFT_create(nfeatures=1500)
        keypoints, desc = sift.detectAndCompute(im, None)
        responses = np.array([k.response for k in keypoints])
        keypoints = np.array([k.pt for k in keypoints]).astype(int)
        desc = np.array(desc)
        detections = np.zeros(im.shape[:2], np.float)
        detections[keypoints[:, 1], keypoints[:, 0]] = responses
        descriptors = np.zeros((im.shape[0], im.shape[1], 128), np.float)
        descriptors[keypoints[:, 1], keypoints[:, 0]] = desc
    elif config['method'] == 'orb':
        orb = cv2.ORB_create(nfeatures=1500)
        keypoints, desc = orb.detectAndCompute(im, None)
        responses = np.array([k.response for k in keypoints])
        keypoints = np.array([k.pt for k in keypoints]).astype(int)
        desc = np.array(desc)
        detections = np.zeros(im.shape[:2], np.float)
        detections[keypoints[:, 1], keypoints[:, 0]] = responses
        descriptors = np.zeros((im.shape[0], im.shape[1], 32), np.float)
        descriptors[keypoints[:, 1], keypoints[:, 0]] = desc
    detections = detections.astype(np.float32)
    descriptors = descriptors.astype(np.float32)
    return detections, descriptors


def SIFT_det(img, img_rgb, visualize=False, nfeatures=2000):
    """
    return: 
        x_all: np [N, 2] (x, y)
        des: np [N, 128] (descriptors)
    """
    img = np.uint8(img)
    sift = cv2.xfeatures2d.SIFT_create(contrastThreshold=1e-05)
    kp, des = sift.detectAndCompute(img, None)
    x_all = np.array([p.pt for p in kp])
    if visualize:
        plt.figure(figsize=(30, 4))
        plt.imshow(img_rgb)
        plt.scatter(x_all[:, 0], x_all[:, 1], s=10, marker='o', c='y')
        plt.show()
    return x_all, des


