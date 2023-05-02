import cv2 as cv
import numpy as np
import matplotlib as mpl
from utils import showImages
from sklearn.mixture import GaussianMixture
from utils import *

mpl.rcParams['image.cmap'] = 'gray'

def preprocess(img):

    # Convert img to grayscale
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

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
    
    return normalizedResult

def segment(img):
    # Convert image to HSV color space
    hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    # Define range of color to segment (in HSV color space)
    lowerColor = np.array([0, 60, 60])
    upperColor = np.array([20, 255, 255])

    # Create a mask based on the defined color range
    mask = cv.inRange(hsv_img, lowerColor, upperColor)

    # Apply the mask to the original image
    result = cv.bitwise_and(img, img, mask=mask)

    # Convert img to grayscale
    result = cv.cvtColor(result, cv.COLOR_BGR2GRAY)

    # Morphological Operations
    
    # Kernel
    kernel = np.ones((7, 7), np.uint8)
    kernel2 = np.ones((3, 3), np.uint8)
    
    # Apply morphological operations
    result = cv.dilate(result, kernel, iterations=5)
    result = cv.erode(result, kernel2, iterations=3)

    # Display result
    # showImages([result], ["Result"])
    
    return result

def graySegment(img):
    # Convert img to grayscale
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    cv.imwrite("../output/gray.jpg", img)
    
    # Filter
    gray_filtered = cv.inRange(img, 190, 255)
    
    cv.imwrite("../output/filtered_gray.jpg", gray_filtered)
    
def gaussianMixture(img):
    # Convert to the YCrCb color space
    img_ycrcb = cv.cvtColor(img, cv.COLOR_BGR2YCrCb)

    # Extract the Cr and Cb channels
    cr = img_ycrcb[:, :, 1]
    cb = img_ycrcb[:, :, 2]

    # Concatenate the Cr and Cb channels
    cr_cb = np.stack((cr, cb), axis=-1)

    # Reshape the data for training the GMM
    data = cr_cb.reshape((-1, 2))

    # Fit a GMM to the data
    gmm = GaussianMixture(n_components=2)
    gmm.fit(data)

    # Classify each pixel as hand or non-hand using the GMM
    labels = gmm.predict(data)
    mask = labels.reshape(cr_cb.shape[:2])

    # Apply the mask to the original image to highlight the hand
    result = cv.bitwise_and(img, img, mask=np.uint8(mask * 255))
    return result

def runSegmentation():
    # img = cv.imread("../input/3.jpeg")
    imgs = readImages("../input/")
    # preprocess(img)
    # result = gaussianMixture(img)
    for i in range(len(imgs)):
        segmentationResult = segment(imgs[i]) 
        cv.imwrite("../output2/result" + str(i) + ".jpeg", segmentationResult)
    # showImages([segmentationResult], ["Result"])

def adaptiveThresholding(img):
    # Convert img to grayscale
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    adaptive_thresh = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2)
    return adaptive_thresh

def test():
    img = cv.imread("../input/1_men (2).JPG")
    result = segment(img) 
    # result = adaptiveThresholding(img)
    cv.imwrite("../output/result.jpeg", result)
    # graySegment(img)

if __name__ == '__main__':
    runSegmentation()
    # test()
    