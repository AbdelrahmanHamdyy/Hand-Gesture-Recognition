import cv2 as cv
from os import listdir
from os.path import isfile, join

def readImages(dataPath):
    files = [f for f in listdir(dataPath) if isfile(join(dataPath, f))]
    x = []
    for fileName in files:
        img = cv.imread(dataPath + "/" + fileName, 0)
        x.append(img)
    return x

def showImages(imgs, labels):
    if len(imgs) != len(labels):
        print("Length of images isn't the same as labels")
        return
    for i in range(len(imgs)):
        cv.imshow(labels[i], imgs[i])
        cv.waitKey(0)