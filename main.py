import cv2
import numpy as np
import matplotlib.pyplot as mlp
from core.preprocessing import resize_image, edge_detection

def main():
  

  image = cv2.imread('data/documento.jpg', cv2.IMREAD_GRAYSCALE)  #Reading the image in grayscale

  if image is None:
    print("Error. The image could not have been found")
    exit()
    
  
  edges_image = edge_detection(image, (5,5), 50, 150)
  
  processed_image, scale_factor = resize_image(edges_image, max_dim=800)

  cv2.imshow("Image", processed_image)  #Showing de image
  cv2.waitKey(0)
  cv2.destroyAllWindows()
  
if __name__ == "__main__":
  main()
  
