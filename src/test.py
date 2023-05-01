import cv2
import numpy as np

# Load the image
img = cv2.imread('../input/3.jpeg')

# Convert the image to grayscale
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply adaptive thresholding to create a mask for the shadows
mask = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)

# Apply morphological operations to remove small shadows and smooth out the edges
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

# Apply color correction to the original image
normalized_img = cv2.normalize(img.astype('float'), None, 0, 1, cv2.NORM_MINMAX)
normalized_img = np.clip(normalized_img * 1.5 - 0.5, 0, 1)
corrected_img = (img.astype('float') + normalized_img - 1) * 255 / np.max(img)

# Apply the mask to remove the shadows from the corrected image
shadowless_img = cv2.bitwise_and(corrected_img, corrected_img, mask=mask)

# Display the original and shadowless images
cv2.imshow('Original Image', img)
cv2.imshow('Shadowless Image', shadowless_img)
cv2.waitKey(0)
cv2.destroyAllWindows()