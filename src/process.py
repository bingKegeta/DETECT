import cv2
import numpy as np
import mediapipe as mp

# Initialize Mediapipe for face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

# Define 3D model points.
model_points = np.array([
    (0.0, 0.0, 0.0),             # Nose tip
    (0.0, -330.0, -65.0),        # Chin
    (-225.0, 170.0, -135.0),     # Left eye left corner
    (225.0, 170.0, -135.0),      # Right eye right corner
    (-150.0, -150.0, -125.0),    # Left Mouth corner
    (150.0, -150.0, -125.0)      # Right mouth corner
], dtype=np.float64)

# Camera internals (Assuming no lens distortion)
def get_camera_matrix(img_w, img_h):
    focal_length = img_w
    center = (img_w / 2, img_h / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float64)
    return camera_matrix

def get_head_pose(landmarks, img_w, img_h):
    """
    Estimate head pose using facial landmarks.

    Args:
        landmarks (list): List of facial landmarks.
        img_w (int): Width of the image.
        img_h (int): Height of the image.

    Returns:
        tuple: Rotation and translation vectors.
    """
    image_points = np.array([
        (landmarks[1].x * img_w, landmarks[1].y * img_h),     # Nose tip
        (landmarks[152].x * img_w, landmarks[152].y * img_h), # Chin
        (landmarks[33].x * img_w, landmarks[33].y * img_h),   # Left eye left corner
        (landmarks[263].x * img_w, landmarks[263].y * img_h), # Right eye right corner
        (landmarks[61].x * img_w, landmarks[61].y * img_h),   # Left Mouth corner
        (landmarks[291].x * img_w, landmarks[291].y * img_h)  # Right mouth corner
    ], dtype=np.float64)

    camera_matrix = get_camera_matrix(img_w, img_h)
    dist_coeffs = np.zeros((4,1))  # Assuming no lens distortion

    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        return None, None

    return rotation_vector, translation_vector

def process_frame(frame, x_data, y_data, apply_affine, calibration_data):
    img_h, img_w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    
    eye_state = 'Open'  # Default state

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Optionally apply affine transform
            if apply_affine:
                affine_matrix = get_affine_transform(face_landmarks.landmark, img_w, img_h)
                frame = cv2.warpAffine(frame, affine_matrix, (img_w, img_h))

            # Head pose estimation
            rotation_vector, translation_vector = get_head_pose(face_landmarks.landmark, img_w, img_h)
            if rotation_vector is not None:
                # Optional: Draw head pose axes
                draw_head_pose(frame, rotation_vector, translation_vector, camera_matrix=get_camera_matrix(img_w, img_h))

            # Calculate EAR
            left_EAR, right_EAR = calculate_EAR(face_landmarks.landmark, img_w, img_h)
            avg_EAR = (left_EAR + right_EAR) / 2.0

            # Define EAR threshold
            EAR_THRESHOLD = 0.21  # You may need to adjust this value

            if avg_EAR < EAR_THRESHOLD:
                eye_state = 'Closed'
            elif EAR_THRESHOLD <= avg_EAR < (EAR_THRESHOLD + 0.05):
                eye_state = 'Squint'
            else:
                eye_state = 'Open'

            # Get iris coordinates
            left_iris_x = face_landmarks.landmark[468].x * img_w
            left_iris_y = face_landmarks.landmark[468].y * img_h
            right_iris_x = face_landmarks.landmark[473].x * img_w
            right_iris_y = face_landmarks.landmark[473].y * img_h

            # Average iris coordinates
            avg_x = (left_iris_x + right_iris_x) / 2
            avg_y = (left_iris_y + right_iris_y) / 2

            # Apply calibration offsets
            if calibration_data:
                # Calculate average calibration point (e.g., center)
                center_calib = calibration_data.get('Center', (0, 0))
                if center_calib != (0, 0):
                    calibrated_x = avg_x - center_calib[0]
                    calibrated_y = avg_y - center_calib[1]
                else:
                    calibrated_x = avg_x
                    calibrated_y = avg_y
            else:
                calibrated_x = avg_x
                calibrated_y = avg_y

            # Normalize gaze coordinates
            norm_x, norm_y = normalize_gaze(calibrated_x, calibrated_y, img_w, img_h)

            # Store the calibrated iris coordinates
            if eye_state == 'Open':
                x_data.append(norm_x)
                y_data.append(norm_y)
            else:
                # Optionally, handle closed or squinted eyes
                # For example, append None or interpolate
                x_data.append(None)
                y_data.append(None)

            # Visual confirmation: draw tracked iris on the frame
            cv2.circle(frame, (int(left_iris_x), int(left_iris_y)), 2, (0, 255, 0), -1)
            cv2.circle(frame, (int(right_iris_x), int(right_iris_y)), 2, (0, 255, 0), -1)

            # Display eye state on the frame
            cv2.putText(frame, f"Eye State: {eye_state}", (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        return frame, (eye_state == 'Open'), (rotation_vector, translation_vector)

    return frame, False, (None, None)

def normalize_gaze(calibrated_x, calibrated_y, img_w, img_h):
    """
    Normalize gaze coordinates based on image width and height.

    Args:
        calibrated_x (float): Calibrated x-coordinate.
        calibrated_y (float): Calibrated y-coordinate.
        img_w (int): Width of the image.
        img_h (int): Height of the image.

    Returns:
        tuple: Normalized (x, y) coordinates ranging from -1 to 1.
    """
    norm_x = (calibrated_x - img_w / 2) / (img_w / 2)
    norm_y = (calibrated_y - img_h / 2) / (img_h / 2)
    return norm_x, norm_y


def draw_head_pose(frame, rotation_vector, translation_vector, camera_matrix):
    """
    Draws head pose axes on the frame for visualization.

    Args:
        frame (np.array): The image frame.
        rotation_vector (np.array): Rotation vector from solvePnP.
        translation_vector (np.array): Translation vector from solvePnP.
        camera_matrix (np.array): Camera matrix.
    """
    axis_length = 100
    axis = np.float32([
        [axis_length,0,0],
        [0,axis_length,0],
        [0,0,axis_length]
    ]).reshape(-1,3)

    dist_coeffs = np.zeros((4,1))  # Assuming no lens distortion

    imgpts, _ = cv2.projectPoints(axis, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
    imgpts = np.int32(imgpts).reshape(-1,2)

    nose_tip = (int(translation_vector[0][0]), int(translation_vector[1][0]))
    frame = cv2.line(frame, tuple(nose_tip), tuple(imgpts[0]), (255,0,0), 3)  # X axis in blue
    frame = cv2.line(frame, tuple(nose_tip), tuple(imgpts[1]), (0,255,0), 3)  # Y axis in green
    frame = cv2.line(frame, tuple(nose_tip), tuple(imgpts[2]), (0,0,255), 3)  # Z axis in red

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

def calculate_EAR(landmarks, img_w, img_h):
    """
    Calculate the Eye Aspect Ratio (EAR) for both eyes.

    Args:
        landmarks (list): Facial landmarks.
        img_w (int): Width of the image.
        img_h (int): Height of the image.

    Returns:
        tuple: (left_EAR, right_EAR)
    """
    # Define landmark indices for left and right eyes
    # Using Mediapipe's 468 and 473 for iris, we need to define points around the eyes
    # Example indices for left eye
    left_eye_indices = [33, 160, 158, 133, 153, 144]
    right_eye_indices = [362, 385, 387, 263, 373, 380]

    def eye_aspect_ratio(eye_points):
        # Compute the euclidean distances between the vertical eye landmarks
        A = np.linalg.norm(eye_points[1] - eye_points[5])
        B = np.linalg.norm(eye_points[2] - eye_points[4])

        # Compute the euclidean distance between the horizontal eye landmarks
        C = np.linalg.norm(eye_points[0] - eye_points[3])

        # Compute EAR
        ear = (A + B) / (2.0 * C)
        return ear

    left_eye = np.array([
        [landmarks[idx].x * img_w, landmarks[idx].y * img_h]
        for idx in left_eye_indices
    ], dtype=np.float32)

    right_eye = np.array([
        [landmarks[idx].x * img_w, landmarks[idx].y * img_h]
        for idx in right_eye_indices
    ], dtype=np.float32)

    left_EAR = eye_aspect_ratio(left_eye)
    right_EAR = eye_aspect_ratio(right_eye)

    return left_EAR, right_EAR

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
