import numpy as np
from skimage.feature import hog, local_binary_pattern
import cv2 as cv


def lbp(img, radius=3, n_points=8):
    """
    Compute Local Binary Pattern (LBP) features for an image.

    Parameters:
        img: 2D numpy array representing the image
        radius: radius of the circular LBP sampling region (default: 3)
        n_points: number of sampling points in the circular region (default: 8)

    Returns:
        2D numpy array representing the LBP features
    """
    lbp_img = local_binary_pattern(img, n_points, radius, "uniform")
    hist, _ = np.histogram(
        lbp_img.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2)
    )
    hist = hist.astype("float")
    hist /= hist.sum() + 1e-7
    return hist


def hog_features(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(3, 3)):
    """
    Compute Histogram of Oriented Gradients (HOG) features for an image.

    Parameters:
        img: 2D numpy array representing the image
        orientations: number of gradient orientation bins (default: 9)
        pixels_per_cell: size of a cell in pixels (default: (8, 8))
        cells_per_block: number of cells in each block (default: (3, 3))

    """
    hog_feats = hog(
        img,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        visualize=False,
        feature_vector=True,
    )
    return hog_feats


# ? for test
# img = cv.imread("../input/1/result0.jpeg", 0)
# cv.imshow("input\1\result", img)
# cv.waitKey(0)

# lbp_feats = lbp(img)
# hog_feats = hog_features(img)

# print(hog_feats)
