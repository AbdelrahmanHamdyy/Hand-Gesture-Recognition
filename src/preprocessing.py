import cv2 as cv
import numpy as np
import matplotlib as mpl
from sklearn.mixture import GaussianMixture
from utils import *

mpl.rcParams['image.cmap'] = 'gray'


def getAvg(img):
    # Convert image to grayscale
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Calculate average pixel intensity
    avg_intensity = cv.mean(img_gray)[0]

    return avg_intensity


def gammaLUT(img):
    avg = getAvg(img)

    # gamma > 1 ---> The image becomes darker
    gamma = 0.7
    if (avg > 150):
        gamma = 1.3

    # Create lookup table
    lookup_table = np.zeros((256, 1), dtype=np.uint8)
    for i in range(256):
        lookup_table[i][0] = 255 * pow(i/255.0, gamma)

    # Apply gamma correction using the lookup table
    img_gamma = cv.LUT(img, lookup_table)

    return img_gamma


def contours(img):
    # Find contours
    contour, hier = cv.findContours(img, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)

    # Draw contours
    for cnt in contour:
        cv.drawContours(img, [cnt], 0, 255, -1)

    return img


def restoreImage(mask, img):
    return cv.bitwise_and(img, img, mask=mask)


def setSideBorders(img, val):
    img[:, 0] = val
    img[:, img.shape[1] - 1] = val
    # img[img.shape[0] - 1, :] = val
    return img


def boundingRect(img):
    # apply binary thresholding to the grayscale image
    _, thresh = cv.threshold(img, 0, 255, cv.THRESH_BINARY)

    # find the contours in the binary image
    contours, _ = cv.findContours(
        thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # find the largest contour by area
    max_contour = max(contours, key=cv.contourArea)

    # create a bounding rectangle around the contour
    x, y, w, h = cv.boundingRect(max_contour)

    return x, y, w, h


def crop(img):
    # Get max contour bounding rectangle vertices
    x, y, w, h = boundingRect(img)

    # crop the image to the bounding rectangle
    crop_img = img[y:y+h, x:x+w]

    return crop_img


def segmentYCbCr(img):
    # Convert image to YCbCr color space
    ycbcr_image = cv.cvtColor(img, cv.COLOR_BGR2YCrCb)

    lower_skin = np.array([0, 135, 75], dtype=np.uint8)
    upper_skin = np.array([255, 180, 125], dtype=np.uint8)

    skin_mask = cv.inRange(ycbcr_image, lower_skin, upper_skin)

    output_image = np.zeros(img.shape, np.uint8)
    output_image[skin_mask == 255] = img[skin_mask == 255]

    # Convert img to grayscale
    output_image = cv.cvtColor(output_image, cv.COLOR_BGR2GRAY)

    return output_image


def preprocess(img):
    # Resize
    img = cv.resize(img, (500, 500))

    # Apply gamma correction to adjust lighting
    img = gammaLUT(img)

    # Segmentation
    segmentedImg = segmentYCbCr(img)

    # Convert original image to grayscale
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Structuring Elements for Morphological Operations
    dilationkernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (17, 17))
    erosionkernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))

    # Erosion to remove noise
    erodedImg = cv.erode(segmentedImg, erosionkernel, iterations=3)

    # Dilation to help fill the inner holes
    dilatedImg = cv.dilate(erodedImg, dilationkernel, iterations=4)

    # Region Filling using Contours
    imgWithContours = contours(dilatedImg)

    # Erosion again to clean the image from outside
    erosionkernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (15, 15))
    erodedImg = cv.erode(imgWithContours, erosionkernel, iterations=4)

    # Apply Mask
    maskedImg = restoreImage(erodedImg, img)

    # Crop image to fit the hand exactly
    croppedImg = crop(maskedImg)

    return croppedImg


def augmentImages(imgs):
    new_imgs = []
    print("Augmenting")
    for j in range(len(imgs)):
        current_image = imgs[j]
        height, width = current_image.shape[:2]
        rotation_matrix = cv.getRotationMatrix2D(
            (width/2, height/2), 45, 0.5)
        rotated_image = cv.warpAffine(
            current_image, rotation_matrix, (width, height))
        fliped_image = cv.flip(current_image, 3)
        new_imgs.append(current_image)
        new_imgs.append(fliped_image)
        new_imgs.append(rotated_image)
    return new_imgs


def removeShadows(img):
    # Convert img to grayscale
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Split image channels
    rgbPlanes = cv.split(img)

    planes = []
    normalizedPlanes = []
    for plane in rgbPlanes:
        dilatedImg = cv.dilate(plane, np.ones((7, 7), np.uint8))
        blurImg = cv.medianBlur(dilatedImg, 21)
        diffImg = 255 - cv.absdiff(plane, blurImg)
        normalizedImg = cv.normalize(
            diffImg, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8UC1)
        planes.append(diffImg)
        normalizedPlanes.append(normalizedImg)

    result = cv.merge(planes)
    normalizedResult = cv.merge(normalizedPlanes)

    # Show images
    showImages([img, result, normalizedResult], [
               "Input", "Result", "Norm Result"])

    return normalizedResult


def graySegment(img):
    # Convert img to grayscale
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    cv.imwrite("../output/gray.jpg", img)

    # Filter
    gray_filtered = cv.inRange(img, 190, 255)

    cv.imwrite("../output/filtered_gray.jpg", gray_filtered)


def adaptiveThresholding(img):
    # Convert img to grayscale
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Apply Adaptive Thresholding
    adaptive_thresh = cv.adaptiveThreshold(
        img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2)
    return adaptive_thresh


def gammaCorrection(img):
    gamma = 1.2

    # Apply Gamma Correction
    gamma_corrected = np.power(img/255.0, gamma)
    gamma_corrected = np.uint8(gamma_corrected*255)

    return gamma_corrected


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


def regionFilling(img):
    # Create a mask with zeros, same size as input image
    mask = np.zeros((img.shape[0] + 2, img.shape[1] + 2), np.uint8)

    # Set the seed point (x,y) where the fill will start
    seed_point = (0, 0)

    # Set the fill color (BGR format)
    fill_color = (0, 255, 0)  # Green

    # Specify the lower and upper color range for the fill
    lower_color = (0, 0, 0)
    upper_color = (10, 10, 10)

    # Perform the flood fill operation
    cv.floodFill(img, mask, seed_point, fill_color, lower_color,
                 upper_color, flags=cv.FLOODFILL_MASK_ONLY)

    return img


def segmentHSV(img):
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

    return result


def runPreprocessing():
    imgs = readImages("../testInput/")
    for i in range(len(imgs)):
        result = preprocess(imgs[i])
        cv.imwrite("../output/result" + str(i) + ".jpeg", result)


if __name__ == '__main__':
    runPreprocessing()
