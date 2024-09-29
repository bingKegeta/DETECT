import cv2
import numpy as np
import mediapipe as mp

# Initialize Mediapipe for face and iris detection
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

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

            # Get iris coordinates
            left_iris_x = face_landmarks.landmark[468].x * img_w
            left_iris_y = face_landmarks.landmark[468].y * img_h
            right_iris_x = face_landmarks.landmark[473].x * img_w
            right_iris_y = face_landmarks.landmark[473].y * img_h

            # Average iris coordinates
            avg_x = (left_iris_x + right_iris_x) / 2
            avg_y = (left_iris_y + right_iris_y) / 2

            # Store the iris coordinates
            x_data.append(avg_x)
            y_data.append(avg_y)

            # Visual confirmation: draw tracked iris on the frame
            cv2.circle(frame, (int(left_iris_x), int(left_iris_y)), 2, (0, 255, 0), -1)
            cv2.circle(frame, (int(right_iris_x), int(right_iris_y)), 2, (0, 255, 0), -1)

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
