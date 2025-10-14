# Import libraries
from scipy.signal import convolve2d
import cv2
import numpy as np

# Load an image (grayscale and color)
image = cv2.imread("./images/lenna.png", 0)

# Create a kernel
kernel = np.random.uniform(-0.5, 1, (3, 3))

# Set needed variables
padding = 1 # Amout of 0s to add around the image
stride = 50 # Step size for moving the kernel

# Funtion to Flip the Kernel
def flipp_kernel(kernel):
    flipped_kernel = np.zeros_like(kernel)
    for i in range(kernel.shape[0]):
        for j in range(kernel.shape[1]):
            flipped_kernel[i, j] = kernel[kernel.shape[0] - 1 - i, kernel.shape[1] - 1 - j]

    return flipped_kernel

# Function to pad the image
def add_padding(image, pad):
    return np.pad(image, ((pad, pad), (pad, pad)), mode='constant', constant_values=0)

# Function to divide the image into multiple subimages, by stride.
def create_stride(image, kernel_size = 3, stride = 1):
    subimages = []

    y, x = np.shape(image)
    cols = int((x - kernel_size) / stride) + 1
    rows = int((y - kernel_size) / stride) + 1
    for j in range(rows):
        j_start, j_end = (j, j+ kernel_size)
        for i in range(cols):
            i_start, i_end = (i, i+ kernel_size)
            subimages.append(image[j_start:j_end, i_start:i_end])

    return subimages



def convolution(image, kernel, padding=0, stride=1):
    """
    Performs a convolution operation on a multi-channel image using a given kernel.

    Parameters:
      image: A 2D array representing the input image (height, width).
      kernel: A 2D array representing the convolution kernel (height, width).
      padding: An integer representing the amount of padding.
      stride: An integer representing the step size for moving the kernel.

    Returns:
      An array representing the output image after applying the convolution operation.
    """
    # Remember to flip the kernel
    # Use the np.pad() function for padding, and implement only one padding mode (e.g., constant)
    # Your code goes here!
    flipped_kernel = flipp_kernel(kernel)
    
    padded_image = np.pad(image, padding)
    subimages = create_stride(padded_image, stride)
    print(subimages[0])

    output = subimages * flipped_kernel

    return output

# Apply your convolution
custom_result = convolution(image, kernel, padding, stride)

# OpenCV's convolution
# opencv_result = cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_REPLICATE)
# cv2.imshow('Filtered Image', custom_result)

cv2.imshow(f'Library Image 0', custom_result[0])

# cv2.imshow('Library Image', opencv_result)
cv2.imshow('Original Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
# Display the image and convolved_image
# Your code to display the images goes here!