import cv2
import mediapipe as mp
import numpy as np
from collections import deque

# Initialize Mediapipe face mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

# Webcam capture
cap = cv2.VideoCapture(0)

# Landmark indices for left and right eye in Mediapipe FaceMesh
LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]

# Deques for smoothing pupil position
left_pupil_queue = deque(maxlen=5)
right_pupil_queue = deque(maxlen=5)

# Helper function to smooth the pupil position using moving average
def smooth_position(queue, new_position):
    queue.append(new_position)
    return np.mean(queue, axis=0).astype(int)

def isolate_eye_region(frame, eye_landmarks):
    # Extract the eye region using the bounding box of the eye landmarks
    x_min = np.min(eye_landmarks[:, 0])
    x_max = np.max(eye_landmarks[:, 0])
    y_min = np.min(eye_landmarks[:, 1])
    y_max = np.max(eye_landmarks[:, 1])

    # Extract the eye region from the frame
    return frame[y_min:y_max, x_min:x_max], (x_min, y_min)

def detect_pupil(eye_region):
    # Convert to grayscale for thresholding
    gray_eye = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
    
    # Enhance contrast using histogram equalization
    gray_eye = cv2.equalizeHist(gray_eye)
    
    # Apply a Gaussian blur to reduce noise
    blurred_eye = cv2.GaussianBlur(gray_eye, (7, 7), 0)
    
    # Adaptive thresholding for more robust pupil detection in varying lighting
    thresholded_eye = cv2.adaptiveThreshold(blurred_eye, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 5)

    # Find contours (the pupil will be the largest contour)
    contours, _ = cv2.findContours(thresholded_eye, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Find the largest contour, assuming it's the pupil
        max_contour = max(contours, key=cv2.contourArea)
        (x, y), radius = cv2.minEnclosingCircle(max_contour)
        return int(x), int(y), int(radius)

    return None

def calculate_gaze(pupil_position, eye_region_width):
    x_ratio = pupil_position[0] / eye_region_width

    if x_ratio < 0.4:
        return 'Right'
    elif x_ratio > 0.6:
        return 'Left'
    else:
        return 'Center'

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame for face landmarks
    result = face_mesh.process(rgb_frame)

    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
            # Draw only eye landmarks, remove full face drawing
            for idx in LEFT_EYE_INDICES + RIGHT_EYE_INDICES:
                landmark = face_landmarks.landmark[idx]
                x, y = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

            # Extract left and right eye landmarks
            left_eye_landmarks = np.array([(int(face_landmarks.landmark[i].x * w),
                                            int(face_landmarks.landmark[i].y * h)) for i in LEFT_EYE_INDICES])
            right_eye_landmarks = np.array([(int(face_landmarks.landmark[i].x * w),
                                             int(face_landmarks.landmark[i].y * h)) for i in RIGHT_EYE_INDICES])

            # Isolate the left and right eye regions
            left_eye_region, left_eye_origin = isolate_eye_region(frame, left_eye_landmarks)
            right_eye_region, right_eye_origin = isolate_eye_region(frame, right_eye_landmarks)

            # Detect pupils in the left and right eyes
            left_pupil = detect_pupil(left_eye_region)
            right_pupil = detect_pupil(right_eye_region)

            if left_pupil:
                # Smooth the pupil position
                left_pupil_x, left_pupil_y = smooth_position(left_pupil_queue, left_pupil[:2])
                
                # Draw the detected pupil on the left eye
                cv2.circle(left_eye_region, (left_pupil_x, left_pupil_y), left_pupil[2], (255, 0, 0), 2)

                # Determine the gaze direction based on pupil position
                left_gaze = calculate_gaze((left_pupil_x, left_pupil_y), left_eye_region.shape[1])
                cv2.putText(frame, f"Left Eye Gaze: {left_gaze}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            if right_pupil:
                # Smooth the pupil position
                right_pupil_x, right_pupil_y = smooth_position(right_pupil_queue, right_pupil[:2])
                
                # Draw the detected pupil on the right eye
                cv2.circle(right_eye_region, (right_pupil_x, right_pupil_y), right_pupil[2], (255, 0, 0), 2)

                # Determine the gaze direction based on pupil position
                right_gaze = calculate_gaze((right_pupil_x, right_pupil_y), right_eye_region.shape[1])
                cv2.putText(frame, f"Right Eye Gaze: {right_gaze}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Display left and right eyes in their positions on the main frame
            cv2.imshow('Eye Tracking', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
