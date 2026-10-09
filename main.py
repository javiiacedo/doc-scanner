import cv2
import numpy as np
import matplotlib.pyplot as mlp
from core.preprocessing import resize_image, edge_detection, find_document_contours
from core.transform import transform
from PIL import Image
import os

points = []   #Corners clicked by the user

def click_event(event, x, y, flags, param):
  if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
    points.append((x, y))
    cv2.circle(param, (x, y), 5, (255,0,0), -1)   #param is display_image
    cv2.imshow("Image", param)

def save_as_pdf(image, path):
  """
  Saves a BGR OpenCV image as a one-page PDF.
  """
  os.makedirs(os.path.dirname(path), exist_ok=True)
  rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)   #OpenCV uses BGR, Pillow uses RGB
  Image.fromarray(rgb).save(path, "PDF", resolution=150.0)


def main():
  

  image = cv2.imread('data/documento2.jpeg')  #Reading the image in bgr

  if image is None:
    print("Error. The image could not have been found")
    exit()
    
  processed_image, scale_factor = resize_image(image, max_dim=800)
  
  display_image = processed_image.copy()    #Where we draw the clicks
  
  cv2.imshow("Image", display_image)  #Showing de image
  cv2.setMouseCallback("Image", click_event, display_image)   #Click the 4 corners with the mouse
  while len(points) < 4:
    if cv2.waitKey(1) == 27:   #ESC to exit
      break
  print("Corners:", points)

  if len(points) == 4:
    pts = np.array(points, dtype='float32') / scale_factor   #Clicks were made on the resized image
    warped = transform(image, pts)
    processed_image2, scale_factor = resize_image(warped, max_dim=800)
    cv2.imshow("Transformed", processed_image2)

    save_as_pdf(warped, "output/scanned.pdf")   #Full resolution image, not the reduced one
    print("PDF saved in output/scanned.pdf")

  cv2.waitKey(0)
  cv2.destroyAllWindows()
  
if __name__ == "__main__":
  main()
  
