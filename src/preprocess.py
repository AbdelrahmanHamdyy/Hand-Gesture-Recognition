import cv2 as cv
import numpy as np
import matplotlib as mpl
from utils import showImages

mpl.rcParams['image.cmap'] = 'gray'

def preprocess():
    # Read image as grayscale
    img = cv.imread("../input/3.jpeg", 0)
    
    # Convert to black & white
    # (thresh, img) = cv.threshold(img, 128, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    
    # Smooth the image using a gaussian filter
    img = cv.GaussianBlur(img, (5, 5), 0)
    
    # Smooth the image using a median filter
    # img = cv.medianBlur(img, 5)
    
    # Remove shadows
    rgbPlanes = cv.split(img)

    planes = []
    normalizedPlanes = []
    for plane in rgbPlanes:
        dilatedImg = cv.dilate(plane, np.ones((7,7), np.uint8))
        blurImg = cv.medianBlur(dilatedImg, 21)
        diffImg = 255 - cv.absdiff(plane, blurImg)
        normalizedImg = cv.normalize(diffImg, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8UC1)
        planes.append(diffImg)
        normalizedPlanes.append(normalizedImg)
    
    result = cv.merge(planes)
    normalizedResult = cv.merge(normalizedPlanes)
    
    # Show images
    showImages([img, result, normalizedResult], ["Input", "Result", "Norm Result"])
    
if __name__ == '__main__':
    preprocess()
    