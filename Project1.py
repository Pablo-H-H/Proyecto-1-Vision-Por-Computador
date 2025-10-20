# Import libraries
from scipy.signal import convolve2d
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load an image (grayscale and color)
image = cv2.imread("./images/lenna.png", 0)

# Create a kernel
kernel = np.ones((3,3))/9

# Set needed variables
padding = 1
stride = 1

### Compare outputs from your convolution implementation with OpenCV and SciPy ###
fig, ax = plt.subplots(1, 4, figsize=(16, 5))

# Your convolution
custom_result = convolution(image, kernel, padding, stride)
ax[0] = plt.imshow(custom_result)

# OpenCV's convolution
opencv_result = cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_REPLICATE)
ax[1] = plt.imshow(opencv_result)

# Scipy's convolution
scipy_result = convolve2d( image, kernel, mode='same')
ax[2] = plt.imshow(scipy_result)

# Display the custom_result, opencv_result and scipy_result
# Your code to display the images goes here!