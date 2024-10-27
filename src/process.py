import math
import cv2
import numpy as np
import mediapipe as mp
from src.utils import HorizontalRegion, VerticalRegion

# Initialize Mediapipe for face and iris detection
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

# Doing Euclidean distance manually since np needs to have landmarks converted to 
# np arrays which has a considerable overhead for our purposes
def distance(landmark1, landmark2):
    return math.sqrt(
        (landmark2.x - landmark1.x)**2 +
        (landmark2.y - landmark1.y)**2 +
        (landmark2.z - landmark1.z)**2
    )

# Process the frame for face and iris detection
def process_frame(frame, x_data, y_data, apply_affine, display, categorize):
    img_h, img_w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Extract landmarks for left eye
            left_eye_indices = [33, 133, 159, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
            left_eye_landmarks = np.array([(face_landmarks.landmark[idx].x * img_w,
                                            face_landmarks.landmark[idx].y * img_h) for idx in left_eye_indices], dtype=np.float32)
            
            # Extract landmarks for right eye
            right_eye_indices = [263, 362, 386, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466]
            right_eye_landmarks = np.array([(face_landmarks.landmark[idx].x * img_w,
                                             face_landmarks.landmark[idx].y * img_h) for idx in right_eye_indices], dtype=np.float32)
            
            # Process each eye separately
            left_eye_img, left_eye_coords = extract_and_normalize_eye(frame, left_eye_landmarks)
            right_eye_img, right_eye_coords = extract_and_normalize_eye(frame, right_eye_landmarks)
            
            # Find iris position in the normalized eye image
            left_iris_pos = find_iris_position(left_eye_img)
            right_iris_pos = find_iris_position(right_eye_img)
            
            # Average the iris positions
            avg_iris_x = (left_iris_pos[0] + right_iris_pos[0]) / 2
            avg_iris_y = (left_iris_pos[1] + right_iris_pos[1]) / 2
            
            # Store the normalized coordinates
            x_data.append(avg_iris_x)
            y_data.append(avg_iris_y)
            
            # Visualization (Optional)
            if display:
                # Draw the left eye region
                for i in range(len(left_eye_landmarks)):
                    cv2.circle(frame, (int(left_eye_landmarks[i][0]), int(left_eye_landmarks[i][1])), 1, (0, 255, 0), -1)
                # Draw the right eye region
                for i in range(len(right_eye_landmarks)):
                    cv2.circle(frame, (int(right_eye_landmarks[i][0]), int(right_eye_landmarks[i][1])), 1, (0, 255, 0), -1)
                
                # Draw iris positions on the normalized eye images (if you display them)
                cv2.circle(left_eye_img, (int(left_iris_pos[0] * left_eye_img.shape[1]), int(left_iris_pos[1] * left_eye_img.shape[0])), 2, (0, 0, 255), -1)
                cv2.circle(right_eye_img, (int(right_iris_pos[0] * right_eye_img.shape[1]), int(right_iris_pos[1] * right_eye_img.shape[0])), 2, (0, 0, 255), -1)
                
                if categorize:
                    region = f"Gaze Direction: {HorizontalRegion(avg_iris_x)}-{VerticalRegion(avg_iris_y)}"
                    cv2.putText(frame, region, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    
            return frame, True

    return frame, False

def extract_and_normalize_eye(frame, eye_landmarks):
    # Compute the center and angle of the eye
    left_corner = eye_landmarks[0]
    right_corner = eye_landmarks[8]  # Adjust index based on landmarks
    eye_center = np.mean(eye_landmarks, axis=0)
    dx = right_corner[0] - left_corner[0]
    dy = right_corner[1] - left_corner[1]
    angle = np.degrees(np.arctan2(dy, dx))
    
    # Compute the affine transformation matrix
    M = cv2.getRotationMatrix2D(tuple(eye_center), angle, 1.0)
    
    # Rotate the image to align the eye horizontally
    img_h, img_w = frame.shape[:2]
    rotated_frame = cv2.warpAffine(frame, M, (img_w, img_h))
    
    # Apply the same transformation to the eye landmarks
    eye_landmarks_rotated = np.array([M @ [x, y, 1] for (x, y) in eye_landmarks])
    
    # Crop the eye region
    x, y, w, h = cv2.boundingRect(eye_landmarks_rotated.astype(np.int32))
    eye_img = rotated_frame[y:y+h, x:x+w]
    
    # Resize the eye image to a standard size
    eye_img_resized = cv2.resize(eye_img, (60, 36))
    
    # Normalize the coordinates of the eye landmarks within the eye image
    eye_landmarks_normalized = (eye_landmarks_rotated - [x, y]) / [w, h]
    
    return eye_img_resized, eye_landmarks_normalized

def find_iris_position(eye_img):
    # Convert to grayscale
    gray_eye = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY)
    
    # Use thresholding to segment the iris
    _, thresh_eye = cv2.threshold(gray_eye, 30, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours
    contours, _ = cv2.findContours(thresh_eye, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Assume the largest contour is the iris
        largest_contour = max(contours, key=cv2.contourArea)
        # Compute the centroid of the iris
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cx = M["m10"] / M["m00"] / eye_img.shape[1]
            cy = M["m01"] / M["m00"] / eye_img.shape[0]
            return (cx, cy)
    # Fallback if iris is not detected
    return (0.5, 0.5)
