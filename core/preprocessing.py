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

def edge_detection(image):
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
    filtered = cv2.GaussianBlur(image, (5, 5), 0)
    _, binary = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def find_document_contours(edges):
    """
    Finds the largest contour in a binary edge map that can be approximated 
    as a quadrilateral (4 vertices).

    Args:
        edges (numpy.ndarray): Binary image (output from Canny).

    Returns:
        numpy.ndarray or None: The coordinates of the 4 vertices of the document,
                               or None if no valid document is found.
    """
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    c0 = max(cnts, key=cv2.contourArea)
    
    hull = cv2.convexHull(c0)
    peri = cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, 0.02 * peri, True)
    
    if len(approx) == 4:
        return approx
    return None
