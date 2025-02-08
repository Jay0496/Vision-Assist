
# Dynamic ROI Object Recognition with Hand Pointing & YOLOv8

This project demonstrates an interactive system that uses hand tracking to detect where a user points and then applies object detection on a dynamically defined Region of Interest (ROI). The detected object's label is then announced via text-to-speech. The system leverages:

- **MediaPipe Hands** for real-time hand landmark detection.
- **YOLOv8** for object detection.
- **pyttsx3** for text-to-speech functionality.
- **OpenCV** for image processing and visualization.


## Dependencies

- Python 3.6 - 3.11
- [OpenCV](https://opencv.org/)
- [MediaPipe](https://mediapipe.dev/)
- [NumPy](https://numpy.org/)
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [pyttsx3](https://pyttsx3.readthedocs.io/)

### Installation

You can install the required packages using pip:

```bash
pip install opencv-python mediapipe numpy ultralytics pyttsx3
```

# How It Works
1. Hand Tracking:

The webcam feed is processed using MediaPipe to extract hand landmarks. The positions of the wrist and index fingertip are used to compute a pointing vector.

2. Dynamic ROI Calculation:

The pointing vector is normalized and scaled (using a tunable scale parameter) to determine the center of the ROI.
A fixed-size ROI (defined by roi_size) is centered at this computed point.
The code ensures that the ROI remains within the frame boundaries.

3. Object Detection with YOLOv8:

Every few frames (controlled by a frame counter), the ROI is passed to the YOLOv8 model.
The model returns detections (bounding boxes and class labels) within the ROI.
The highest-confidence detection is selected, and its label is adjusted to frame coordinates for visualization.

4. Text-to-Speech:

If a new object is detected (different from the last announced object), the system uses pyttsx3 to announce the object label.
This prevents repetitive announcements and keeps the feedback dynamic.

5. Visualization:

The application displays hand landmarks, the pointing vector, the ROI rectangle, detection bounding boxes, and labels on the output frame.
An arrow is drawn to show the pointing direction.


# Troubleshooting
Empty ROI Error:
If you encounter an empty ROI, verify that the ROI calculation is correct and adjust the scale and roi_size parameters.

No Hand Detected:
Ensure good lighting conditions and that your hand is clearly visible to the camera.

Performance Issues:
If the system is slow, increase the frame interval to reduce the frequency of YOLO detections.

