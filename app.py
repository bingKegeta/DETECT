from flask import Flask, request, jsonify
import cv2
import numpy as np
from gaze_detect import process_frame  # Import your gaze detection function
import mediapipe as mp

app = Flask(__name__)

# Initialize MediaPipe for face mesh and drawing utilities
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

# Route to process video frames from the frontend
@app.route('/process-frame', methods=['POST'])
def process_frame_route():
    # Get the image file from the request
    file = request.files['frame'].read()

    # Convert the image to a format usable by OpenCV
    npimg = np.frombuffer(file, np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    # Initialize lists to store gaze coordinates
    x_data, y_data = [], []

    # Call your gaze detection function (from gaze_detect.py)
    processed_frame = process_frame(frame, x_data, y_data)

    # Assuming your gaze_detect.py processes gaze coordinates and returns them
    gaze_data = {
        'x_data': x_data,
        'y_data': y_data,
    }

    # Send back the processed gaze coordinates
    return jsonify(gaze_data)

if __name__ == '__main__':
    app.run(debug=True)
