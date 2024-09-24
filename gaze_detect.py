import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import mediapipe as mp
import time

# Initialize Mediapipe for face and iris detection
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

# Savitzky-Golay filter parameters
WINDOW_LENGTH = 9  # Must be odd
POLYORDER = 2

# CLI argument parsing
def parse_args():
    parser = argparse.ArgumentParser(description="Gaze and pupil tracking with smoothing and graphing options.")
    
    parser.add_argument('--source', type=str, required=True, choices=['webcam', 'image', 'video'],
                        help="Source type for gaze detection: 'webcam', 'image', or 'video'.")
    
    parser.add_argument('--path', type=str, required=False, help="Path to image or video file.")
    
    parser.add_argument('--graph', action='store_true', help="Show a real-time graph of gaze coordinates.")
    
    return parser.parse_args()

# Function to apply Savitzky-Golay filter for smoothing
def apply_savgol_filter(data):
    if len(data) >= WINDOW_LENGTH:
        return savgol_filter(data, WINDOW_LENGTH, POLYORDER)
    return data

def get_affine_transform(landmarks, img_w, img_h):
    """
    Compute the affine transformation matrix to align the face based on stable facial landmarks.
    We use the left eye, right eye, and nose bridge to stabilize head position.
    """
    # Choose three landmarks: left eye, right eye, and nose bridge
    left_eye = np.array([landmarks[33].x * img_w, landmarks[33].y * img_h])
    right_eye = np.array([landmarks[263].x * img_w, landmarks[263].y * img_h])
    nose_bridge = np.array([landmarks[1].x * img_w, landmarks[1].y * img_h])

    # Define the reference points for alignment (centered, aligned face)
    ref_left_eye = np.array([img_w * 0.3, img_h * 0.4])
    ref_right_eye = np.array([img_w * 0.7, img_h * 0.4])
    ref_nose_bridge = np.array([img_w * 0.5, img_h * 0.6])

    src_points = np.array([left_eye, right_eye, nose_bridge], dtype=np.float32)
    dst_points = np.array([ref_left_eye, ref_right_eye, ref_nose_bridge], dtype=np.float32)

    # Compute the affine transformation matrix
    affine_matrix = cv2.getAffineTransform(src_points, dst_points)
    
    return affine_matrix

def process_frame(frame, x_data, y_data):
    img_h, img_w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Get the affine transformation matrix to stabilize the face
            affine_matrix = get_affine_transform(face_landmarks.landmark, img_w, img_h)

            # Apply the affine transformation to the frame
            stabilized_frame = cv2.warpAffine(frame, affine_matrix, (img_w, img_h))

            # After stabilization, extract the iris coordinates
            left_iris_x = face_landmarks.landmark[468].x * img_w
            left_iris_y = face_landmarks.landmark[468].y * img_h
            right_iris_x = face_landmarks.landmark[473].x * img_w
            right_iris_y = face_landmarks.landmark[473].y * img_h

            # Apply the same affine transformation to the iris coordinates
            left_iris_transformed = np.dot(affine_matrix, np.array([left_iris_x, left_iris_y, 1]))
            right_iris_transformed = np.dot(affine_matrix, np.array([right_iris_x, right_iris_y, 1]))

            # Average the iris positions for better stability
            x_data.append((left_iris_transformed[0] + right_iris_transformed[0]) / 2)
            y_data.append((left_iris_transformed[1] + right_iris_transformed[1]) / 2)

            # Draw the stabilized iris positions on the frame
            cv2.circle(stabilized_frame, (int(left_iris_transformed[0]), int(left_iris_transformed[1])), 2, (0, 255, 0), -1)
            cv2.circle(stabilized_frame, (int(right_iris_transformed[0]), int(right_iris_transformed[1])), 2, (0, 255, 0), -1)

    return stabilized_frame

# Main function to handle video/webcam or image processing
def main():
    args = parse_args()

    # Initialize video capture based on CLI input
    if args.source == 'webcam':
        cap = cv2.VideoCapture(0)
    elif args.source in ['image', 'video']:
        if not args.path:
            print("You must provide a --path to the image or video file.")
            return
        cap = cv2.VideoCapture(args.path)
    else:
        print("Invalid source type provided. Use --source with 'webcam', 'image', or 'video'.")
        return

    # Initialize data arrays to store gaze coordinates over time
    x_data = []
    y_data = []
    time_data = []

    if args.graph:
        # Initialize the plot
        plt.ion()  # Turn on interactive plotting
        fig, ax = plt.subplots()
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Gaze Coordinates')
        ax.set_title('Gaze Tracking Over Time')

    start_time = time.time()

    # If the source is an image, process it directly
    if args.source == 'image':
        ret, frame = cap.read()
        if ret:
            frame = process_frame(frame, x_data, y_data)
            cv2.imshow('Gaze Tracking', frame)
            if args.graph:
                plt.pause(0.01)
            cv2.waitKey(0)
        else:
            print("Unable to read image.")
        cap.release()
        cv2.destroyAllWindows()
        if args.graph:
            plt.close()
        return

    # For video or webcam, process frames in a loop
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Calculate the elapsed time
        current_time = time.time() - start_time
        time_data.append(current_time)

        # Process the frame to detect and smooth gaze coordinates
        frame = process_frame(frame, x_data, y_data)

        # Apply Savitzky-Golay filter for smoothing x and y coordinates
        x_data_smoothed = apply_savgol_filter(x_data)
        y_data_smoothed = apply_savgol_filter(y_data)

        # Display the frame with gaze landmarks (iris detection shown)
        cv2.imshow('Gaze Tracking', frame)

        # Plot the smoothed gaze coordinates over time if graph is enabled
        if args.graph:
            ax.clear()
            ax.plot(time_data, x_data_smoothed, label="X Coordinate (Left-Right)")
            ax.plot(time_data, y_data_smoothed, label="Y Coordinate (Up-Down)")
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Gaze Coordinates')
            ax.set_title('Gaze Tracking Over Time')
            ax.legend()
            plt.pause(0.01)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    if args.graph:
        plt.close()

if __name__ == "__main__":
    main()
