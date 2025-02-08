import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO

# Initialize the YOLOv8 model (you can choose a larger model if desired)
model = YOLO('yolov8m.pt')  # or 'yolov8s.pt' for better accuracy

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)  

with mp_hands.Hands(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # Process the frame with MediaPipe Hands
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks on the frame
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Get wrist and index finger tip landmarks
                wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

                # Convert normalized coordinates to pixel coordinates
                wrist_px = np.array([int(wrist.x * w), int(wrist.y * h)])
                index_tip_px = np.array([int(index_tip.x * w), int(index_tip.y * h)])

                # Calculate pointing vector
                pointing_vector = index_tip_px - wrist_px

                if np.linalg.norm(pointing_vector) == 0:
                    continue  # Skip if pointing vector is zero

                pointing_vector_normalized = pointing_vector / np.linalg.norm(pointing_vector)

                # Initial scale for ROI
                scale = 100  # Adjust based on your setup
                roi_center = index_tip_px + (pointing_vector_normalized * scale).astype(int)

                # Ensure ROI center is within frame boundaries
                roi_center[0] = np.clip(roi_center[0], 0, w - 1)
                roi_center[1] = np.clip(roi_center[1], 0, h - 1)

                # Initial ROI size
                roi_size = 150
                x1 = int(max(roi_center[0] - roi_size // 2, 0))
                y1 = int(max(roi_center[1] - roi_size // 2, 0))
                x2 = int(min(roi_center[0] + roi_size // 2, w))
                y2 = int(min(roi_center[1] + roi_size // 2, h))

                roi = frame[y1:y2, x1:x2]

                if roi.size == 0:
                    continue  # Skip if ROI is empty

                # Perform object detection on the ROI
                results_yolo = model.predict(source=roi, conf=0.5, imgsz=320, verbose=False)
                detections = results_yolo[0]

                if len(detections.boxes) > 0:
                    # Get the detection with the highest confidence
                    max_conf_idx = np.argmax(detections.boxes.conf.cpu().numpy())
                    detection_box = detections.boxes.xyxy[max_conf_idx].cpu().numpy().astype(int)
                    class_id = int(detections.boxes.cls[max_conf_idx])
                    detected_label = model.names[class_id]

                    # Coordinates relative to ROI; adjust to frame coordinates
                    dx1, dy1, dx2, dy2 = detection_box
                    dx1 += x1
                    dy1 += y1
                    dx2 += x1
                    dy2 += y1

                    # Update ROI to the new dynamic ROI based on detection
                    x1_new = max(dx1 - roi_size // 4, 0)
                    y1_new = max(dy1 - roi_size // 4, 0)
                    x2_new = min(dx2 + roi_size // 4, w)
                    y2_new = min(dy2 + roi_size // 4, h)

                    # Draw the adjusted ROI rectangle
                    cv2.rectangle(frame, (x1_new, y1_new), (x2_new, y2_new), (255, 0, 0), 2)

                    # Optionally, perform additional detection on the expanded ROI
                    # roi_expanded = frame[y1_new:y2_new, x1_new:x2_new]
                    # Optional further processing...

                    # Draw the detection bounding box
                    cv2.rectangle(frame, (dx1, dy1), (dx2, dy2), (0, 255, 0), 2)
                    cv2.putText(frame, detected_label, (dx1, dy1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    # If no detection, you can choose to expand the ROI or handle it differently
                    pass

                # Visualize the pointing vector
                cv2.arrowedLine(frame, (wrist_px[0], wrist_px[1]), (index_tip_px[0], index_tip_px[1]),
                                (0, 0, 255), 2)
        else:
            print("No hands detected.")

        cv2.imshow("Dynamic ROI Object Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
