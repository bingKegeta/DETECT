import cv2
import mediapipe as mp
import argparse
import sys

# Initialize MediaPipe FaceMesh and Drawing modules
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Landmark indices for left and right irises
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]
PUPILS = [468, 473]  # Right pupil (468), Left pupil (473)

def process_frame(frame):
    """Process a frame (from webcam or image) to detect face and iris, then display the result."""
    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame using the FaceMesh model
    results = face_mesh.process(frame_rgb)

    # Variables to store pupil coordinates for display in the corners
    left_pupil_coords = (0, 0)
    right_pupil_coords = (0, 0)

    # If face landmarks are detected
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Draw the face mesh landmarks on the frame
            mp_drawing.draw_landmarks(
                frame,
                face_landmarks,
                mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())

            # Get the coordinates of the pupils (landmarks 468 and 473)
            for idx, pupil_landmark in enumerate(PUPILS):
                landmark = face_landmarks.landmark[pupil_landmark]
                x = int(landmark.x * w)
                y = int(landmark.y * h)

                # Save the pupil coordinates for the left and right eyes
                if idx == 0:  # Right pupil
                    right_pupil_coords = (x, y)
                elif idx == 1:  # Left pupil
                    left_pupil_coords = (x, y)

    # Display the gaze coordinates in the corners
    cv2.putText(frame, f"Right Gaze: {right_pupil_coords}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(frame, f"Left Gaze: {left_pupil_coords}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    
    return frame

def webcam_mode():
    """Capture frames from webcam and process them."""
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        processed_frame = process_frame(frame)
        cv2.imshow('Gaze Tracking with MediaPipe', processed_frame)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def image_mode(image_path):
    """Process a single image and display the result."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image at {image_path}")
        sys.exit(1)

    processed_image = process_frame(image)
    cv2.imshow('Gaze Tracking with MediaPipe', processed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def video_mode(video_path):
    """Process a video file or camera stream frame by frame and display the result."""
    # Open the video file or camera stream
    capture = cv2.VideoCapture(video_path)
    
    # Check if the video opened successfully
    if not capture.isOpened():
        print(f"Error: Could not open video at {video_path}")
        sys.exit(1)
    
    # Loop to read and process frames until the video ends or interrupted
    while True:
        # Read a frame from the video
        ret, frame = capture.read()
        
        # If the frame could not be read, we have reached the end of the video
        if not ret:
            print("End of video or cannot fetch the frame.")
            break
        
        # Process the current frame
        processed_frame = process_frame(frame)
        
        # Display the processed frame
        cv2.imshow('Gaze Tracking with MediaPipe', processed_frame)
        
        # Wait for 1 ms and check if the user wants to quit by pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release video capture and close all OpenCV windows
    capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Track face and iris using MediaPipe.")
    parser.add_argument('--webcam', action='store_true', help='Use webcam for face tracking')
    parser.add_argument('--image', type=str, help='Path to the image file')
    parser.add_argument('--video', type=str, help='Path to video file')

    args = parser.parse_args()

    # Initialize face mesh model
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # Handle different modes
    if args.webcam:
        webcam_mode()
    elif args.image:
        image_mode(args.image)
    elif args.video:
        video_mode(args.video)
    else:
        print("Error: You must provide either --webcam or --image <path>")
        sys.exit(1)

    # Clean up the face mesh model
    face_mesh.close()
