"""Testing file (not sorted yet)

"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
from numpy.linalg import inv

from utils.utils import inv_warp_image_batch
from utils.draw import plot_imgs
from utils.utils import pltImshow

path = (
    '/home/yoyee/Documents/deepSfm/logs/superpoint_hpatches_pretrained/predictions/')
for i in range(10):
    data = np.load(path + str(i) + '.npz')

    H = data['homography']
    img1 = data['image'][:, :, np.newaxis]
    img2 = data['warped_image'][:, :, np.newaxis]
    warped_img1 = cv2.warpPerspective(img1, H, (img1.shape[1], img1.shape[0]))

    img1 = np.concatenate([img1, img1, img1], axis=2)
    warped_img1 = np.stack([warped_img1, warped_img1, warped_img1], axis=2)
    img2 = np.concatenate([img2, img2, img2], axis=2)
    plot_imgs([img1, img2, warped_img1], titles=['img1', 'img2', 'warped_img1'], dpi=200)
    plt.savefig('test' + str(i) + '.png')
