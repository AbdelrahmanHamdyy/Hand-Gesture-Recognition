import cv2 as cv
import numpy as np
from utils import readImages
from skimage.transform import resize
from skimage.feature import hog
from utils import *
from preprocess import *

INPUT_PATH = "../../../Dataset_0-5/men"
featuresDict = {}
def getFeatures(x_train):
    features = []
    for img in x_train:
        img=segment(img)
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        fd = hog(resize(img, (128*4, 64*4)), orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualize=False)
        features.append(fd)
        
    return np.array(features)
    

def saveFeatures():
    for i in range (6):
        print("***************************")
        print(i)
        print("***************************")
        INPUT_PATH = "../../../Dataset_0-5/men/"+str(i)
        imgs = readImages(INPUT_PATH)
        features=getFeatures(imgs)
        
        featuresDict[str(i)] = features

   

def run():
    saveFeatures()
    saveToExcel(featuresDict,"../output/output.xlsx")
    
if __name__ == '__main__':
    run()
    
    
