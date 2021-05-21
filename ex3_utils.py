from typing import List

import cv2
import numpy as np
import matplotlib.pyplot as plt


def opticalFlow(im1: np.ndarray, im2: np.ndarray, step_size=10, win_size=5) -> (np.ndarray, np.ndarray):
    """
    Given two images, returns the Translation from im1 to im2
    :param im1: Image 1
    :param im2: Image 2
    :param step_size: The image sample size:
    :param win_size: The optical flow window size (odd number)
    :return: Original points [[x,y]...],+* [[dU,dV]...] for each points
    """
    kernel_x = np.array([[-1, 0, 1]])
    kernel_y = kernel_x.T

    # window_size is odd, all the pixels with offset in between [-w, w] are inside the window
    w_ = int(win_size / 2)

    # Implement Lucas Kanade Algorithm
    # for each point, calculate I_x, I_y, I_t
    fx_drive = cv2.filter2D(im2, -1, kernel_x, borderType=cv2.BORDER_REPLICATE)
    fy_drive = cv2.filter2D(im2, -1, kernel_y, borderType=cv2.BORDER_REPLICATE)
    ft_drive = im2 - im1

    originalPoints = []
    dU_dV = []
    for i in range(w_, im1.shape[0] - w_ + 1, step_size):
        for j in range(w_, im1.shape[1] - w_ + 1, step_size):
            Ix = fx_drive[i - w_:i + w_, j - w_:j + w_].flatten()
            Iy = fy_drive[i - w_:i + w_, j - w_:j + w_].flatten()
            It = ft_drive[i - w_:i + w_, j - w_:j + w_].flatten()
            AtA_ = [[(Ix * Ix).sum(), (Ix * Iy).sum()],
                    [(Ix * Iy).sum(), (Iy * Iy).sum()]]
            lam_ = np.linalg.eigvals(AtA_)
            lam2_ = np.min(lam_)
            lam1_ = np.max(lam_)
            if lam2_ <= 1 or lam1_ / lam2_ >= 100:
                continue
            Atb_ = [[-(Ix * It).sum()], [-(Iy * It).sum()]]
            v_ = np.linalg.inv(AtA_) @ Atb_
            dU_dV.append(v_)
            originalPoints.append([j, i])
    return np.array(originalPoints), np.array(dU_dV)


def laplaceianReduce(img: np.ndarray, levels: int = 4) -> List[np.ndarray]:
    """
    Creates a Laplacian pyramid
    :param img: Original image
    :param levels: Pyramid depth
    :return: Laplacian Pyramid (list of images)
    """
    gaussianPyramid = gaussianPyr(img, levels)
    laplacian_top = gaussianPyramid[-1]

    laplacian_pyr = [laplacian_top]
    for i in range(levels - 1, 0, -1):
        expand = gaussExpand(gaussianPyramid[i], gaussianKernel() * 4)
        # expand = cv2.pyrUp(gaussianPyramid[i])
        laplacian_i = cv2.subtract(gaussianPyramid[i - 1], expand)
        laplacian_pyr.append(laplacian_i)
    laplacian_pyr.reverse()
    return laplacian_pyr


def laplaceianExpand(lap_pyr: List[np.ndarray]) -> np.ndarray:
    """
    Resotrs the original image from a laplacian pyramid
    :param lap_pyr: Laplacian Pyramid
    :return: Original image
    """
    levels = len(lap_pyr)
    lapTop = lap_pyr[- 1]

    # go through the list from end to start
    for i in range(levels - 1, 0, -1):
        # size = (lap_pyr[i - 1].shape[1], lap_pyr[i - 1].shape[0])
        laplacian_expanded = cv2.pyrUp(lapTop)
        lapTop = cv2.add(lap_pyr[i - 1], laplacian_expanded)
    return lapTop


def gaussianKernel():
    """
    Kernel size : 5 X 5
    sigma : 0.3 * ((kernelSize - 1) * 0.5 - 1) + 0.8
    :return: 2D Gaussian kernel
    """
    sigma = 0.3 * ((5 - 1) * 0.5 - 1) + 0.8
    ax = np.linspace(-2., 2., 5)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sigma))
    return kernel / kernel.sum()


def resizeImg(img, levels):
    """
    Each level in the pyramids, the image shape is cut in half.
    so for x levels, crop the initial image to:
    (2^x) * (int)(IMG_size/2^x)
    :param img:
    :param levels:
    :return:
    """
    n = 2 ** levels
    width, height = n * int(img.shape[1] / n), n * int(img.shape[0] / n)
    img = cv2.resize(img, (width, height))
    img = img.astype(np.float64)
    return img


def reduceImg(img):
    blurImg = cv2.filter2D(img, -1, gaussianKernel(), borderType=cv2.BORDER_REPLICATE)
    img = blurImg[::2, ::2]
    return img


def gaussianPyr(img: np.ndarray, levels: int = 4) -> List[np.ndarray]:
    """
    Creates a Gaussian Pyramid
    :param img: Original image
    :param levels: Pyramid depth
    :return: Gaussian pyramid (list of images)
    """
    gaussianPyramid = []

    # creat the first level (original img) for the pyramid
    img = resizeImg(img.copy(), levels)
    img_i = img.copy()
    gaussianPyramid.append(img_i)

    # Creates a Gaussian Pyramid
    for i in range(1, levels):
        img_i = reduceImg(img_i)
        # img_i = cv2.pyrDown(img_i)
        gaussianPyramid.append(img_i)
    return gaussianPyramid


def gaussExpand(img: np.ndarray, gs_k: np.ndarray) -> np.ndarray:
    """
    Expands a Gaussian pyramid level one step up
    :param img: Pyramid image at a certain level
    :param gs_k: The kernel to use in expanding
    :return: The expanded level
    """
    x, y = img.shape[0], img.shape[1]
    if len(img.shape) == 3:  # RGB
        shape = (x * 2, y * 2, 3)
    else:  # GRAY
        shape = (x * 2, y * 2)
    preImg = np.zeros(shape)
    preImg[::2, ::2] = img
    return cv2.filter2D(preImg, -1, gs_k, borderType=cv2.BORDER_REPLICATE)


def resizeByMask(img: np.ndarray, mask: np.ndarray) -> (np.ndarray, np.ndarray):
    height, width = mask.shape[0], mask.shape[1]
    img = cv2.resize(img, (width, height))
    return img


def pyrBlend(img_1: np.ndarray, img_2: np.ndarray, mask: np.ndarray, levels: int) -> (np.ndarray, np.ndarray):
    """
    Blends two images using PyramidBlend method
    :param img_1: Image 1
    :param img_2: Image 2
    :param mask: Blend mask
    :param levels: Pyramid depth
    :return: (Naive blend, Blended Image)
    """
    img_1, img_2 = resizeByMask(img_1, mask), resizeByMask(img_2, mask)

    # Pyramid Blending
    lapImg_1 = laplaceianReduce(img_1, levels)
    lapImg_2 = laplaceianReduce(img_2, levels)
    gaussMask = gaussianPyr(mask, levels)

    n = levels - 1
    pyramidBlend = gaussMask[n] * lapImg_1[n] + (1 - gaussMask[n]) * lapImg_2[n]
    for i in range(n - 1, -1, -1):
        # upScale = gaussExpand(pyramidBlend, gaussianKernel() * 4)
        upScale = cv2.pyrUp(pyramidBlend)
        pyramidBlend = upScale + gaussMask[i] * lapImg_1[i] + (1 - gaussMask[i]) * lapImg_2[i]

    # Naive Blending
    naiveBlend = mask * img_1 + (1 - mask) * img_2
    naiveBlend = cv2.resize(naiveBlend, (pyramidBlend.shape[1], pyramidBlend.shape[0]))

    return naiveBlend, pyramidBlend
