import cv2 as cv
import numpy as np
from utils import readImages
from skimage.transform import resize
from skimage.feature import hog

INPUT_PATH = "../input/"

def getFeatures(x_train):
    features = []
    for img in x_train:
        fd = hog(resize(img, (128*4, 64*4)), orientations=9, pixels_per_cell=(14, 14),
                 cells_per_block=(1, 1), visualize=False)
        features.append(fd)

    return np.array(features)
    
def run():
    imgs = readImages(INPUT_PATH)
    result = getFeatures(imgs)
    print(result)
    
if __name__ == '__main__':
    run()
    
    
