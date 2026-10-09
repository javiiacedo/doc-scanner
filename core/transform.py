import cv2
import numpy as np

def order_corners(pts):
    """
    With this function we are going to order the 4 corners of the documents: [top-left, top-right, bottom-right, bottom-left]

    Args:
        pts: 4 corners coordinates

    Returns:
        A numpy array with the 4 corners ordered.
    """

    ordered = np.zeros((4,2), dtype='float32')  #Create an empty matrix of 4 rows and 2 columns that will contain new coordinates
    pts = pts.reshape(4,2)

    s = pts.sum(axis=1)   # x+y on each point
    ordered[0] = pts[np.argmin(s)]   #Top left
    ordered[2] = pts[np.argmax(s)]   #Bottom right

    diff = np.diff(pts, axis=1)
    ordered[1] = pts[np.argmin(diff)]   #Top right
    ordered[3] = pts[np.argmax(diff)]   #Bottom left

    return ordered

def transform(image, pts):
    """
    In this function we are going to calculate the size of the document, generate the perspective matrix and return the image with the new perspective
    Args:
        image: image that we are going to transform
        pts: 4 ordered corners coordinates

    Returns:
        Image with a new perspective
    """

    coords = order_corners(pts)
    tl, tr, br, bl = coords

    #Maximum width
    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    width = max(int(width_a), int(width_b))

    #Maximum height
    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    height = max(int(height_a), int(height_b))

    dst = np.array([[0,0], [width-1,0], [width-1, height-1], [0, height-1]], dtype='float32')

    M = cv2.getPerspectiveTransform(coords, dst)
    warped = cv2.warpPerspective(image, M, (width, height))

    return warped




    



    