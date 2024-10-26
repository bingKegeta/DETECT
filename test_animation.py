# test_animation.py

from src.animation import create_gaze_animation

# Sample gaze positions (normalized between -1 and 1)
eye_positions = [
    (0.0, 0.0),
    (0.1, 0.1),
    (0.2, 0.15),
    (0.3, 0.2),
    (0.4, 0.25),
    (0.5, 0.3),
    (0.6, 0.35),
    (0.7, 0.4),
    (0.6, 0.35),
    (0.5, 0.3),
    (0.4, 0.25),
    (0.3, 0.2),
    (0.2, 0.15),
    (0.1, 0.1),
    (0.0, 0.0)
]

frame_width = 640
frame_height = 480
fps = 10  # Frames per second for testing

create_gaze_animation(eye_positions, frame_width, frame_height, fps)
