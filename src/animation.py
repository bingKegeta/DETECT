# src/animation.py

from PyQt5 import QtWidgets, QtGui, QtCore
import os
import sys

class GazeAnimation(QtWidgets.QGraphicsView):
    def __init__(self, eye_positions, frame_width, frame_height, fps=30, save_dir='frames'):
        super().__init__()
        self.eye_positions = eye_positions
        self.fps = fps
        self.current_frame = 0
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.save_dir = save_dir

        # Create directory for frames if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)

        # Set up the scene
        self.scene = QtWidgets.QGraphicsScene()
        self.setScene(self.scene)
        self.setFixedSize(frame_width, frame_height)

        # Calculate padding and eye dimensions
        x_values = [x for x, y in eye_positions if x is not None]
        y_values = [y for x, y in eye_positions if y is not None]
        
        min_x, max_x = min(x_values), max(x_values)
        min_y, max_y = min(y_values), max(y_values)
        padding_x = (max_x - min_x) * 0.1 if (max_x - min_x) != 0 else 1
        padding_y = (max_y - min_y) * 0.1 if (max_y - min_y) != 0 else 1

        ax_min_x, ax_max_x = min_x - padding_x, max_x + padding_x
        ax_min_y, ax_max_y = min_y - padding_y, max_y + padding_y
        self.setSceneRect(ax_min_x, ax_min_y, ax_max_x - ax_min_x, ax_max_y - ax_min_y)

        # Artificial eye setup
        center_x = (ax_min_x + ax_max_x) / 2
        center_y = (ax_min_y + ax_max_y) / 2
        radius = max((ax_max_x - ax_min_x), (ax_max_y - ax_min_y)) / 2 + max(padding_x, padding_y)
        
        # Draw the artificial eye
        self.eye = self.scene.addEllipse(
            center_x - radius, center_y - radius, 2 * radius, 2 * radius,
            pen=QtGui.QPen(QtCore.Qt.black, 2),
            brush=QtGui.QBrush(QtCore.Qt.white)
        )

        # Gaze point setup
        self.gaze_point = QtWidgets.QGraphicsEllipseItem(-7.5, -7.5, 15, 15)
        self.gaze_point.setBrush(QtGui.QBrush(QtCore.Qt.blue))
        self.scene.addItem(self.gaze_point)
        
        # Timer for animation
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(1000 / fps)

    def update_frame(self):
        if self.current_frame < len(self.eye_positions):
            x, y = self.eye_positions[self.current_frame]
            if x is not None and y is not None:
                self.gaze_point.setPos(x, y)
            else:
                self.gaze_point.setPos(self.scene.sceneRect().center())
            
            # Capture and save the current frame as an image
            self.capture_frame()

            self.current_frame += 1
        else:
            self.timer.stop()

    def capture_frame(self):
        # Capture the frame from the QGraphicsView
        image = self.grab()
        frame_path = os.path.join(self.save_dir, f'frame_{self.current_frame:04d}.png')
        image.save(frame_path, 'PNG')


def create_gaze_animation(eye_positions, frame_width, frame_height, fps=30, save_path='animation.mp4'):
    app = QtWidgets.QApplication(sys.argv)
    animation_view = GazeAnimation(eye_positions, frame_width, frame_height, fps)
    animation_view.show()

    app.exec_()

    # Instructions to compile the frames into a video using ffmpeg or similar tool
    print("\nFrames have been saved in the 'frames' directory.")
    print("To compile them into a video, you can use the following command in your terminal:")
    print(f"ffmpeg -r {fps} -i frames/frame_%04d.png -vcodec libx264 -y {save_path}")
