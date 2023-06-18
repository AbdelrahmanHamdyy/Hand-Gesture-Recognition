import numpy as np
from skimage.transform import resize
from skimage.feature import hog, local_binary_pattern
from sklearn.decomposition import PCA
from utils import *
from preprocessing import *


DATA = "../Dataset_0-5/"
SAMPLE = "../Sample/"

featuresDict = {}


def getFeatures(imgs):
    features = []
    for img in imgs:
        # Preprocessing
        img = preprocess(img)

        # HOG
        fd = hog(resize(img, (64*2, 128*2)), orientations=9,
                 pixels_per_cell=(16, 16), cells_per_block=(2, 2), visualize=False)
        features.append(fd)

    return features


def saveFeatures():
    x = []
    y = []
    for i in range(6):
        INPUT_PATH = DATA + str(i)
        imgs = readImages(INPUT_PATH)
        imgs = augmentImages(imgs)
        features = getFeatures(imgs)
        featuresDict[str(i)] = features
        x = x + features
        y = y + ([str(i)] * len(features))

    return x, y, featuresDict


def applyPCA(features):
    pca = PCA(n_components=172)  # keep the top N principal components
    pca.fit(features)
    featuresPCA = pca.transform(features)
    return featuresPCA


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


def sift(img):
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # keypoints
    sift = cv.xfeatures2d.SIFT_create()
    keypoints_1, descriptors_1 = sift.detectAndCompute(img, None)

    res = cv.drawKeypoints(gray_img, keypoints_1, img)
    return res
