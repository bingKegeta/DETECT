import cv2
import numpy as np
import mediapipe as mp
from src.utils import HorizontalRegion, VerticalRegion

# Initialize Mediapipe for face and iris detection
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# Define MediaPipe landmark indices for iris centers and nose tip
LEFT_IRIS_CENTER = 468
LEFT_EYE_CORNER = 33
RIGHT_EYE_CORNER = 263
RIGHT_IRIS_CENTER = 473
NOSE_TIP = 4

# Process the frame for face and iris detection
def process_frame(frame, x_data, y_data, apply_affine, display, categorize):
    img_h, img_w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Align the face to a frontal pose if apply_affine is True
            if apply_affine:
                frame, face_landmarks = align_face(frame, face_landmarks, img_w, img_h)
            
            # Get normalized iris position
            avg_iris_x, avg_iris_y = get_normalized_iris_position(face_landmarks, img_w, img_h)
            
            # Store the normalized coordinates
            x_data.append(avg_iris_x)
            y_data.append(avg_iris_y)
            
            # Visualization (Optional)
            if display:                
                # Optionally, draw all landmarks for verification
                for idx, lm in enumerate(face_landmarks.landmark):
                    x = int(lm.x * img_w)
                    y = int(lm.y * img_h)
                    cv2.circle(frame, (x, y), 1, (255, 0, 0), -1)
                
                if categorize:
                    region = f"Gaze Direction: {HorizontalRegion(avg_iris_x)}-{VerticalRegion(avg_iris_y)}"
                    cv2.putText(frame, region, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            # Since we're processing one face, return after the first
            return frame, True

    return frame, False

def get_normalized_iris_position(face_landmarks, img_w, img_h):
    """
    Computes the normalized iris position relative to the nose tip.

    Args:
        face_landmarks (mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList):
            The facial landmarks detected by MediaPipe.
        img_w (int): Width of the image.
        img_h (int): Height of the image.

    Returns:
        (float, float): Normalized (x, y) coordinates between 0 and 1.
    """
    # Extract iris centers
    left_iris = face_landmarks.landmark[LEFT_IRIS_CENTER]
    right_iris = face_landmarks.landmark[RIGHT_IRIS_CENTER]
    
    # Compute average iris center
    iris_x = (left_iris.x + right_iris.x) / 2
    iris_y = (left_iris.y + right_iris.y) / 2
    
    # Extract nose tip
    nose = face_landmarks.landmark[NOSE_TIP]
    nose_x = nose.x
    nose_y = nose.y
    
    # Compute relative position
    rel_x = iris_x - nose_x
    rel_y = nose_y - iris_y
    
    # Normalize relative to inter-ocular distance (distance between eyes)
    left_eye = face_landmarks.landmark[LEFT_EYE_CORNER]
    right_eye = face_landmarks.landmark[RIGHT_EYE_CORNER]
    inter_ocular_distance = np.linalg.norm(
        np.array([left_eye.x, left_eye.y]) - np.array([right_eye.x, right_eye.y])
    )
    
    norm_x = rel_x / inter_ocular_distance
    norm_y = (rel_y / inter_ocular_distance) # Manual adjustment to set consistent range
    
    return (norm_x, norm_y)

def align_face(frame, face_landmarks, img_w, img_h):
    """
    Aligns the face in the frame to a frontal pose based on eye positions.

    Args:
        frame: The original image frame.
        face_landmarks: Detected facial landmarks from MediaPipe.
        img_w: Width of the image.
        img_h: Height of the image.

    Returns:
        Aligned frame and updated face_landmarks.
    """

    # Extract eye centers
    left_iris = face_landmarks.landmark[LEFT_IRIS_CENTER]
    right_iris = face_landmarks.landmark[RIGHT_IRIS_CENTER]

    # Compute eye centers in pixel coordinates
    left_eye_center = np.array([left_iris.x * img_w, left_iris.y * img_h])
    right_eye_center = np.array([right_iris.x * img_w, right_iris.y * img_h])

    # Compute the angle between the eye centers
    delta_x = right_eye_center[0] - left_eye_center[0]
    delta_y = right_eye_center[1] - left_eye_center[1]
    angle = np.degrees(np.arctan2(delta_y, delta_x))

    # Desired distance between eyes
    desired_dist = 100  # pixels, adjust as needed

    # Compute scale factor
    current_dist = np.linalg.norm(right_eye_center - left_eye_center)
    if current_dist == 0:
        current_dist = 1e-6  # Prevent division by zero
    scale = desired_dist / current_dist

    # Compute center between eyes
    eyes_center = (left_eye_center + right_eye_center) / 2

    # Compute the affine transformation matrix
    M = cv2.getRotationMatrix2D(tuple(eyes_center), angle, scale)

    # Update the translation component of the matrix
    desired_center = np.array([img_w / 2, img_h / 2])
    M[0, 2] += desired_center[0] - eyes_center[0]
    M[1, 2] += desired_center[1] - eyes_center[1]

    # Apply affine transformation
    aligned_frame = cv2.warpAffine(frame, M, (img_w, img_h))

    # Transform all landmarks
    for i, landmark in enumerate(face_landmarks.landmark):
        point = np.array([landmark.x * img_w, landmark.y * img_h, 1])
        transformed_point = M.dot(point)

        # Update the landmark values
        face_landmarks.landmark[i].x = transformed_point[0] / img_w
        face_landmarks.landmark[i].y = transformed_point[1] / img_h

    return aligned_frame, face_landmarks