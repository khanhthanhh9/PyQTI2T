import cv2
import numpy as np


import pandas as pd

def bilateral_filter(image, diameter=9, sigma_color=75, sigma_space=75):
    # Apply the bilateral filter to the image
    filtered_image = cv2.bilateralFilter(image, diameter, sigma_color, sigma_space)
    
    return filtered_image



def enhance_dim_text(image):


    # Create a CLAHE object and apply it to the image
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(image)

    # Return the enhanced image
    return enhanced

def adaptive_threshold(image):
    # Get the average color of the four border pixels
    top = np.mean(image[0])
    bottom = np.mean(image[-1])
    left = np.mean(image[:, 0])
    right = np.mean(image[:, -1])

    # Determine the type of thresholding based on the average color
    if top + bottom + left + right > 510:
        # Use binary thresholding for light background
        thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    else:
        # Use binary inverse thresholding for dark background
        thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    return thresh

def to_grayscale(image):
        # Convert the image to grayscale
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return grayscale_image
    
def morphological_transformations(image, iterations=1):
    # Create a structuring element
    kernel = np.ones((1, 1), np.uint8)
    
    # Dilate the image
    dilated_image = cv2.dilate(image, kernel, iterations=iterations)
    
    # Erode the image
    eroded_image = cv2.erode(dilated_image, kernel, iterations=iterations)
    
    return eroded_image

def scale_dpi(image, target_dpi=300):
    # Get the current DPI of the image
    current_dpi = image.shape[0] / float(image.shape[1])
    
    # Calculate the scaling factor
    scaling_factor = target_dpi / current_dpi
    
    # Resize the image with the scaling factor
    new_size = (int(image.shape[1] * scaling_factor), int(image.shape[0] * scaling_factor))
    resized_image = cv2.resize(image, new_size)
    
    # Save the resized image with the new DPI
    retval, buffer = cv2.imencode('.jpg', resized_image, [int(cv2.IMWRITE_JPEG_QUALITY), 100, int(cv2.IMWRITE_JPEG_PROGRESSIVE), 1])
    output_image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    
    return output_image
    
def add_bilateral_noise(img, d=9, sigmaColor=75, sigmaSpace=75):
    """
    Adds bilateral noise to the input image.

    Parameters:
    img (numpy.ndarray): Input image.
    d (int): Diameter of each pixel neighborhood that is used during filtering (default=9).
    sigmaColor (float): Filter sigma in the color space (default=75).
    sigmaSpace (float): Filter sigma in the coordinate space (default=75).

    Returns:
    numpy.ndarray: Noisy image.
    """
    # Convert the image to float32
    img = np.float32(img)

    # Add bilateral noise to the image
    noisy_img = cv2.bilateralFilter(img, d, sigmaColor, sigmaSpace)

    # Clip the pixel values to the valid range
    noisy_img = np.clip(noisy_img, 0, 255)

    # Convert the image back to uint8
    noisy_img = np.uint8(noisy_img)

    return noisy_img
    
def resize_image(image):
    # Get the original size of the image
    height, width = image.shape[:2]
    
    # Calculate the new size of the image
    new_height, new_width = int(height * 2.4), int(width * 2.4)
    
    # Resize the image using OpenCV's resize function
    resized_image = cv2.resize(image, (new_width, new_height))
    
    return resized_image

def preprocess_image(img):
        # Convert image to np format
        img = np.array(img)
        # Convert the image to grayscale
        img = to_grayscale(img)
        # Resize the image
        img = resize_image(img)

        return img

