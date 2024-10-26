import cv2
import numpy as np
import time
import mediapipe as mp

# Initialize Mediapipe for face and iris detection
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

def draw_calibration_points(frame, points):
    for point in points:
        cv2.circle(frame, (int(point[0]), int(point[1])), 10, (0, 0, 255), -1)
        cv2.putText(frame, point[2], (int(point[0]) - 20, int(point[1]) - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

def run_calibration(cap, calibration_points, duration=5):
    """
    Runs the calibration process by guiding the user to look at specified points.

    Args:
        cap (cv2.VideoCapture): Video capture object.
        calibration_points (list): List of tuples containing (x, y, label) for calibration.
        duration (int): Duration in seconds for each calibration point.

    Returns:
        dict: Calibration data containing average iris positions for each point.
    """
    calibration_data = {}
    for point in calibration_points:
        label = point[2]
        print(f"Please look at the {label} point.")
        start_time = time.time()
        iris_positions = []

        while time.time() - start_time < duration:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame during calibration.")
                continue

            img_h, img_w, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)

            # Check if the webcam feed was successful
            if not results:
                print("Error: Could not start webcam.")
                break

            # Draw the calibration point on the frame
            cv2.circle(frame, (int(point[0]), int(point[1])), 10, (0, 0, 255), -1)
            cv2.putText(frame, label, (int(point[0]) - 20, int(point[1]) - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # Overlay smaller instructions for better readability
            overlay_text = f"Look at the {label} point. Align your head."
            cv2.putText(frame, overlay_text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # Show the frame in a window named 'Calibration'
            cv2.imshow('Calibration', frame)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Get iris coordinates
                    left_iris_x = face_landmarks.landmark[468].x * img_w
                    left_iris_y = face_landmarks.landmark[468].y * img_h
                    right_iris_x = face_landmarks.landmark[473].x * img_w
                    right_iris_y = face_landmarks.landmark[473].y * img_h

                    # Average iris coordinates
                    avg_x = (left_iris_x + right_iris_x) / 2
                    avg_y = (left_iris_y + right_iris_y) / 2

                    iris_positions.append((avg_x, avg_y))

            # Exit if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Calibration interrupted by user.")
                break

        # Average the iris positions for each calibration point
        if iris_positions:
            avg_iris_x = np.mean([pos[0] for pos in iris_positions])
            avg_iris_y = np.mean([pos[1] for pos in iris_positions])
            calibration_data[label] = (avg_iris_x, avg_iris_y)
            print(f"Calibration point '{label}': ({avg_iris_x:.2f}, {avg_iris_y:.2f})")
        else:
            print(f"No iris data collected for '{label}'.")

    cv2.destroyWindow('Calibration')
    return calibration_data
