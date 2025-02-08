
# Dynamic ROI Object Recognition with Hand Pointing & YOLOv8

This project demonstrates an interactive system that uses hand tracking to detect where a user is pointing and then applies object detection on a dynamically defined Region of Interest (ROI). The detected object's label is then announced via text-to-speech. The system leverages:

- **MediaPipe Hands** for real-time hand landmark detection.
- **YOLOv8** for object detection.
- **pyttsx3** for text-to-speech functionality.
- **OpenCV** for image processing and visualization.

## Features

- **Hand Tracking**: Detects hand landmarks using MediaPipe.
- **Dynamic ROI Calculation**: Computes an ROI based on the direction the user is pointing (using wrist and index fingertip landmarks).
- **YOLOv8 Object Detection**: Performs object detection on the ROI every few frames for optimization.
- **Text-to-Speech**: Announces the detected object label when it changes.
- **Optimized Processing**: YOLO detection is run only once every N frames (e.g., every 5 frames) to improve performance.

## Dependencies

- Python 3.6+
- [OpenCV](https://opencv.org/)
- [MediaPipe](https://mediapipe.dev/)
- [NumPy](https://numpy.org/)
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [pyttsx3](https://pyttsx3.readthedocs.io/)

### Installation

You can install the required packages using pip:

```bash
pip install opencv-python mediapipe numpy ultralytics pyttsx3
