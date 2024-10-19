import cv2
import mediapipe as mp
import numpy as np
import argparse
from collections import deque

# Initialize Mediapipe face mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

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

def calculate_gaze(pupil_position, eye_region_width, eye_region_height):
    x_ratio = pupil_position[0] / eye_region_width
    y_ratio = pupil_position[1] / eye_region_height

    # Determine horizontal gaze direction (Left, Right, Center)
    if x_ratio < 0.4:
        horizontal_gaze = 'Left'
    elif x_ratio > 0.6:
        horizontal_gaze = 'Right'
    else:
        horizontal_gaze = 'Center'

    # Determine vertical gaze direction (Up, Down, Center)
    if y_ratio < 0.4:
        vertical_gaze = 'Up'
    elif y_ratio > 0.6:
        vertical_gaze = 'Down'
    else:
        vertical_gaze = 'Center'

    return horizontal_gaze, vertical_gaze

def process_frame(frame):
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame for face landmarks
    result = face_mesh.process(rgb_frame)

    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
            # Draw only eye landmarks, remove full face drawing
            LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
            RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]

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
                left_gaze_h, left_gaze_v = calculate_gaze((left_pupil_x, left_pupil_y), left_eye_region.shape[1], left_eye_region.shape[0])
                cv2.putText(frame, f"Left Eye Gaze: {left_gaze_h}-{left_gaze_v}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            if right_pupil:
                # Smooth the pupil position
                right_pupil_x, right_pupil_y = smooth_position(right_pupil_queue, right_pupil[:2])
                
                # Draw the detected pupil on the right eye
                cv2.circle(right_eye_region, (right_pupil_x, right_pupil_y), right_pupil[2], (255, 0, 0), 2)

                # Determine the gaze direction based on pupil position
                right_gaze_h, right_gaze_v = calculate_gaze((right_pupil_x, right_pupil_y), right_eye_region.shape[1], right_eye_region.shape[0])
                cv2.putText(frame, f"Right Eye Gaze: {right_gaze_h}-{right_gaze_v}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return frame

def main():
    # Argument parsing
    parser = argparse.ArgumentParser(description="Gaze Tracking using Eye and Pupil Detection")
    parser.add_argument('--source', choices=['webcam', 'image', 'video'], required=True, help="Source type: webcam, image, or video")
    parser.add_argument('--path', type=str, help="Path to image or video file (required if source is image or video)")
    args = parser.parse_args()

    windowName = "Gaze-Tracking"
    
    # Webcam mode
    if args.source == 'webcam':
        cv2.namedWindow(windowName)
        cap = cv2.VideoCapture(0)
        cap.set(3, 1080)
        cap.set(4, 1920)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = process_frame(frame)
            cv2.imshow(windowName, frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    # Image mode
    elif args.source == 'image':
        if not args.path:
            print("Error: You must provide a valid image path using --path")
            return

        frame = cv2.imread(args.path)
        cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_LINEAR)
        if frame is None:
            print(f"Error: Could not read image at {args.path}")
            return

        frame = process_frame(frame)
        cv2.imshow('Eye Tracking', frame)
        # cv2.set(4, 1920)
        cv2.waitKey(0)  # Wait until a key is pressed
        cv2.destroyAllWindows()

    # Video mode
    elif args.source == 'video':
        if not args.path:
            print("Error: You must provide a valid video path using --path")
            return

        cap = cv2.VideoCapture(args.path)

        # Get the width and height of the frames
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # Get the FPS of the input video
        fps = cap.get(cv2.CAP_PROP_FPS)  

        # Define the codec and create VideoWriter object (filename, codec, fps, frame size)
        out = cv2.VideoWriter('output_processed_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = process_frame(frame)
            cv2.imshow('Eye Tracking', frame)

            # Write the processed frame to the output video file
            out.write(frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
