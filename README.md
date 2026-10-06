# Intelligent Document Scanner (Computer Vision)

In this project, the core objective is to build a document scanner capable of scanning a physical document to convert an image into PDF format. It is developed in Python, primarily leveraging the NumPy and OpenCV libraries for image and video processing.

## 🚀 Features
*   Real-time video capture.
*   Application of smoothing filters, the Canny algorithm for edge detection, and Harris corner detector / NCC for keypoint extraction.
*   Adaptive geometric detection, perspective correction (Homography), and automatic vertex alignment.

## 🧠 Technical Pipeline (Architecture)
Briefly explains the 4 mathematical stages each frame undergoes:
1.  **Capture and I/O**: Reading from the video buffer using `cv2.VideoCapture`.
2.  **Spatial Preprocessing**: In this initial stage, we convert the image to grayscale to easily locate the boundaries of the target document. Next, a Gaussian filter is applied to remove potential image noise, followed by edge detection using the Canny algorithm. Key functions used: `cv2.cvtColor`, `cv2.GaussianBlur`, and `cv2.Canny`.
3.  **Geometry Extraction**: Contour detection, polygonal curve approximation (Douglas-Peucker algorithm) to simplify vertices, and heuristic filtering to isolate the largest quadrangular polygon.
4.  **Perspective Transformation**: Spatial ordering of the 4 detected corners and computation of the homography matrix (`cv2.warpPerspective`) to extract the rectified, top-down view.

## ⚙️ Requirements & Installation
Follow these instructions and commands to set up and run the system locally:

- Python installed and a functional webcam.
- Clone this repository.
- Create a virtual environment: `python -m venv venv`
- Activate the environment (in CMD, as PowerShell may restrict script execution due to execution policy permissions): `.\venv\Scripts\activate` (run this inside your project root directory).
- Install NumPy and OpenCV inside the newly created virtual environment.

`git clone https://github.com/javiiacedo/doc-scanner.git`
`cd Documents/doc-scanner`
`python -m venv venv`
`.\venv\Scripts\activate`
`pip install opencv-python numpy`

## 📂 Project's structure

*   `main.py`:
*   `io/camera_stream.py`:
*   `core/`: 

## 🎮 Usage

`python main.py`
