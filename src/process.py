import math
import cv2
import numpy as np
import mediapipe as mp

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

def process_frame(frame, x_data, y_data, apply_affine):
    img_h, img_w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Optionally apply affine transform
            if apply_affine:
                affine_matrix = get_affine_transform(face_landmarks.landmark, img_w, img_h)
                frame = cv2.warpAffine(frame, affine_matrix, (img_w, img_h))

            # Store the left eye landmarks
            left_y_top = face_landmarks.landmark[470]
            left_y_bot = face_landmarks.landmark[472]
            left = face_landmarks.landmark[468]
            left_x_lft = face_landmarks.landmark[33]
            left_x_rgt = face_landmarks.landmark[133]
            
            # Store the right eye landmarks
            right_y_top = face_landmarks.landmark[475]
            right_y_bot = face_landmarks.landmark[477]
            right = face_landmarks.landmark[473]
            right_x_lft = face_landmarks.landmark[36]
            right_x_rgt = face_landmarks.landmark[263]
            
            
            # Get iris coordinates
            left_iris_x = distance(left, left_x_rgt) / distance(left_x_lft, left_x_rgt)
            left_iris_y = distance(left, left_y_bot) / distance(left_y_top, left_y_bot)
            right_iris_x = distance(right, right_x_lft) / distance(right_x_lft, right_x_rgt)
            right_iris_y = distance(left, left_y_bot) / distance(right_y_top, right_y_bot)

            # Average iris coordinates
            avg_x = (left_iris_x + right_iris_x) / 2
            avg_y = (left_iris_y + right_iris_y) / 2

            # Store the iris coordinates
            x_data.append(avg_x)
            y_data.append(avg_y)

            # Visual confirmation: draw tracked iris on the frame
            cv2.circle(frame, (int(left.x * img_w), int(left.y * img_h)), 2, (0, 255, 0), -1)
            cv2.circle(frame, (int(right.x * img_w), int(right.y * img_h)), 2, (0, 255, 0), -1)

        return frame, True

    return frame, False

def get_affine_transform(landmarks, img_w, img_h):
    left_eye = np.array([landmarks[33].x * img_w, landmarks[33].y * img_h])
    right_eye = np.array([landmarks[263].x * img_w, landmarks[263].y * img_h])
    nose_bridge = np.array([landmarks[1].x * img_w, landmarks[1].y * img_h])

    ref_left_eye = np.array([img_w * 0.3, img_h * 0.4])
    ref_right_eye = np.array([img_w * 0.7, img_h * 0.4])
    ref_nose_bridge = np.array([img_w * 0.5, img_h * 0.6])

    src_points = np.array([left_eye, right_eye, nose_bridge], dtype=np.float32)
    dst_points = np.array([ref_left_eye, ref_right_eye, ref_nose_bridge], dtype=np.float32)

    return cv2.getAffineTransform(src_points, dst_points)
