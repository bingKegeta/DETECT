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
PUPILS = [468, 473]

def process_frame(frame):
    """Process a frame (from webcam or image) to detect face and iris, then display the result."""
    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame using the FaceMesh model
    results = face_mesh.process(frame_rgb)

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

            # Draw bounding box around the face
            x_min, y_min = w, h
            x_max, y_max = 0, 0
            for landmark in face_landmarks.landmark:
                x, y = int(landmark.x * w), int(landmark.y * h)
                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x_max, x)
                y_max = max(y_max, y)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            # Highlight the irises
            for iris_landmark in LEFT_IRIS + RIGHT_IRIS + PUPILS:
                landmark = face_landmarks.landmark[iris_landmark]
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)
    
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
        cv2.imshow('MediaPipe FaceMesh with Iris Tracking', processed_frame)

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
    cv2.imshow('MediaPipe FaceMesh with Iris Tracking', processed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Track face and iris using MediaPipe.")
    parser.add_argument('--webcam', action='store_true', help='Use webcam for face tracking')
    parser.add_argument('--image', type=str, help='Path to the image file')

    args = parser.parse_args()

    # Initialize face mesh model
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # Handle different modes
    if args.webcam:
        webcam_mode()
    elif args.image:
        image_mode(args.image)
    else:
        print("Error: You must provide either --webcam or --image <path>")
        sys.exit(1)

    # Clean up the face mesh model
    face_mesh.close()
