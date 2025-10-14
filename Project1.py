# Import libraries
from scipy.signal import convolve2d
import cv2
import numpy as np

# Load an image (grayscale and color)
image = cv2.imread("./images/lenna.png", 0)

# Create a kernel
kernel = np.random.random((3,3))

# Set needed variables
padding = 1 # Amout of 0s to add around the image
stride = 1 # Step size for moving the kernel

# Funtion to Flip the Kernel
def flipp_kernel(kernel):
    kernel = cv2.flip(kernel, -1)
    return kernel

# Function to pad the image
def padding(image, pad):
    padded_image = np.pad(image, pad)
    return padded_image

# Function to divide the image into multiple subimages, by stride.
def stride(image, stride = 1):
    subimages = []

    y, x = np.shape(image)
    cols = x / stride
    rows = y / stride 
    for j in rows:
        j_start, j_end = (j * stride, (j+1) * stride)
        for i in cols:
            i_start, i_end = (i * stride, (i+1) * stride)
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

      
    output = cv2.filter2D(padded_image, -1, kernel=flipped_kernel)

    return output


# Apply your convolution
custom_result = convolution(image, kernel, padding, stride)
cv2.imshow('Filtered Image', custom_result)
cv2.waitKey(0)
cv2.destroyAllWindows()
# Display the image and convolved_image
# Your code to display the images goes here!