import cv2 as cv
import numpy as np
from os import listdir
from os.path import isfile, join
from skimage.transform import resize
from skimage.feature import hog

INPUT_PATH = "../input/"

def prepareData(dataPath, files):
    x = []
    for fileName in files:
        img = cv.imread(dataPath + "/" + fileName, 0)
        x.append(img)
    return x

def train(x_train):
    list_hog_fd = []
    for img in x_train:
        fd = hog(resize(img, (128*4, 64*4)), orientations=9, pixels_per_cell=(14, 14),
                 cells_per_block=(1, 1), visualize=False)
        list_hog_fd.append(fd)

    x_train = np.array(list_hog_fd)
    return x_train
    
def run():
    input = [f for f in listdir(INPUT_PATH) if isfile(join(INPUT_PATH, f))]
    imgs = prepareData(INPUT_PATH, input)
    result = train(imgs)
    print(result)
    
if __name__ == '__main__':
    run()
    
    
