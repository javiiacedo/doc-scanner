import cv2
import numpy as np
import matplotlib.pyplot as mlp
from core.preprocessing import resize_image, edge_detection, find_document_contours

def main():
  

  image = cv2.imread('data/documento2.jpeg', cv2.IMREAD_GRAYSCALE)  #Reading the image in grayscale

  if image is None:
    print("Error. The image could not have been found")
    exit()
    
  processed_image, scale_factor = resize_image(image, max_dim=800)
  
  edges_image = edge_detection(processed_image, 50, 150)   #Find edges
  
  contours = find_document_contours(edges_image)  #Find contours of the document
  
  display_image = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2BGR)   #Convert the image from grayscale to bgr
  
  if contours is not None:
    cv2.drawContours(display_image, [contours], -1, (0,255,0), 2)
    for v in contours:
      x, y = v[0]
      cv2.circle(display_image, (x,y), 5, (0,0,255), -1)
  else:
    print("No contour of 4 points have been found")

  cv2.imshow("Image", display_image)  #Showing de image
  cv2.waitKey(0)
  cv2.destroyAllWindows()
  
if __name__ == "__main__":
  main()
  
