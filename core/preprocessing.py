import cv2
import numpy as np

def resize_image(image, max_dim=800):
    """
    Rescales the image so that its largest dimension does not exceed max_dim,
    preserving the original aspect ratio.

    Args:
        image (numpy.ndarray): Input image tensor or 2D matrix.
        max_dim (int): Maximum size allowed for the largest spatial axis.

    Returns:
        tuple: (scaled_image, scale_factor)
            - scaled_image (numpy.ndarray): Resized image ready for processing.
            - scale_factor (float): The scaling ratio applied (needed for homography projection).
    """
    height, width = image.shape[:2]   #Height and width

    # Skip scaling if image is already within boundary
    if max(height, width) <= max_dim:
        return image.copy(), 1.0

    if width >= height:
        scale_factor = max_dim / float(width)
        new_width = max_dim
        new_height = int(height * scale_factor)
    else:
        scale_factor = max_dim / float(height)
        new_height = max_dim
        new_width = int(width * scale_factor)

    scaled_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return scaled_image, scale_factor

def edge_detection(image, kernel, lower_threshold, upper_threshold):
    """
        Apply gaussian blur for smoothing the image, reducing noise and Canny algorithm for detecting edges
    
        Args:
            image (numpy.ndarray): Input image tensor or 2D matrix.
            kernel: Matrix of a determined size and aperture for gaussian filter.
            lower_threshold: Lower boundary for the hysteresis thresholding in Canny.
            upper_threshold: Upper boundary for the hysteresis thresholding in Canny.
    
        Returns:
            A binary picture where white pixels (255) represent edges
        """
    blurred_image = cv2.GaussianBlur(image, kernel, sigmaX=0)  #Smooth the image to reduce noise
    edges = cv2.Canny(blurred_image, lower_threshold, upper_threshold)
    return edges