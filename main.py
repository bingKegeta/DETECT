import re
import cv2
import time
import numpy as np
import sys
import os
import pyqtgraph as pg
from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QHBoxLayout, QWidget, QPushButton
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt
from src.args import load_config
from src.process import process_frame, process_baseline_video
from src.export import export_csv, export_graph, export_features_csv
from src.graph import plot_final_graphs
from src.utils import load_features_data
from src.hmm import HiddenMarkovModel
import warnings

# Suppress specific warning from protobuf
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf.symbol_database")

class WebcamWindow(QWidget):
    """Window to display webcam feed."""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Webcam")
        self.video_label = QLabel(self)
        self.video_label.setScaledContents(True)
        self.video_label.setFixedSize(640, 480)

        layout = QVBoxLayout()
        layout.addWidget(self.video_label)
        self.setLayout(layout)

    def update_video(self, frame):
        """Convert the frame to QImage and update the video QLabel."""
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        q_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(q_image))


class GazeTrackingApp(QWidget):
    def __init__(self, config, fps=30):
        super().__init__()
        self.config = config
        self.initUI()
        self.paused = False
        self.cap = None
        self.webcam_window = None

        # Arrays for storing gaze coords + time
        self.fps = fps
        self.frame_count = 0
        self.time_data = []
        self.x_data = []
        self.y_data = []
        self.deception_data = []

    def initUI(self):
        main_layout = QVBoxLayout(self)
        self.setWindowTitle("Graphs")

        self.win = pg.GraphicsLayoutWidget(show=True, title="Gaze Tracking")
        main_layout.addWidget(self.win, stretch=1)

        # Plot 1: X vs time
        self.ax_x = self.win.addPlot(title='X Coordinate Over Time')
        self.ax_x.setLabel('left', 'X Coordinate')
        self.ax_x.setLabel('bottom', 'Time (s)')
        self.x_curve = self.ax_x.plot(pen='r')

        # Plot 2: Y vs time
        self.ax_y = self.win.addPlot(title='Y Coordinate Over Time', row=1, col=0)
        self.ax_y.setLabel('left', 'Y Coordinate')
        self.ax_y.setLabel('bottom', 'Time (s)')
        self.y_curve = self.ax_y.plot(pen='b')

        # Plot 3: 2D scatter
        self.scatter_ax = self.win.addPlot(title='2D Gaze Points Over Time', row=2, col=0)
        self.scatter = pg.ScatterPlotItem(pen='g')
        self.scatter_ax.addItem(self.scatter)

        # Play/Pause Button
        self.play_pause_button = QPushButton("Pause")
        self.play_pause_button.clicked.connect(self.toggle_pause)
        self.play_pause_button.setStyleSheet("background-color: lightblue; color: white;")
        main_layout.addWidget(self.play_pause_button)

        # Close Button
        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.close_app)
        self.close_button.setStyleSheet("background-color: red; color: white;")
        main_layout.addWidget(self.close_button)

    def toggle_pause(self):
        self.paused = not self.paused
        self.play_pause_button.setText("Play" if self.paused else "Pause")

    def close_app(self):
        """Stop webcam, export data, then close."""
        try:
            # Instead of storing final features, we'll store raw x,y,time in CSV
            # so we can reprocess them later.
            # We must 2-trim time/x/y to align with how we do it in feature extraction
            # but the user asked to store the entire raw sequence, so let's just store them all.
            # The user can re-trim inside prepare_features if needed.

            # We'll store the entire arrays in the session CSV
            export_features_csv(self.x_data, self.y_data, self.time_data,
                                os.path.join(self.config["export_dir"], "session.csv"))

            # Now, load them back & produce features so we can get deception_data
            # This step ensures we produce the final graphs for this run
            x_arr, y_arr, t_arr = load_features_data(os.path.join(self.config["export_dir"], "session.csv"))
            features = self.hmm.prepare_features(x_arr, y_arr, t_arr)

            raw_prob = self.hmm.predict_proba(features)
            self.deception_data = raw_prob[:, 1]

            # For graph export, we need time_data aligned with features => time_data[2..]
            # But the user wants the final CSV to reflect the alignment also, so let's slice them:
            aligned_time = t_arr[2:]
            aligned_x = x_arr[2:]
            aligned_y = y_arr[2:]

            if len(aligned_time) != len(self.deception_data):
                raise ValueError(f"Mismatch: aligned_time({len(aligned_time)}) vs deception_data({len(self.deception_data)})")

            if self.config['export']['csv'] or self.config['export']['graph']:
                if not os.path.exists(self.config['export_dir']):
                    os.makedirs(self.config['export_dir'], exist_ok=True)

            # Export final CSV with deception, time, x, y
            if self.config['export']['csv']:
                csv_path = os.path.join(self.config['export_dir'], "gaze_data.csv")
                # We'll keep the existing signature of export_csv but adapt to store x,y
                # in place of variance,acceleration. We'll handle that in export.py changes
                # so that we have time, x, y, deception
                from src.export import export_csv
                export_csv(features, aligned_time, self.deception_data, csv_path)
                print(f"CSV file saved to: {csv_path}")

            # Export final graph
            if self.config['export']['graph']:
                from src.export import export_graph
                graph_path = os.path.join(self.config['export_dir'], "final_comprehensive_plots.png")
                export_graph(features, aligned_time, self.deception_data, graph_path)

        except Exception as e:
            print(f"Error during export: {e}")

        # Clean up
        if self.cap:
            self.cap.release()
        if self.webcam_window:
            self.webcam_window.close()
        self.close()
        cv2.destroyAllWindows()
        QApplication.quit()
        print("Application closed successfully.")

    def cleanup(self):
        if self.cap:
            self.cap.release()
        if self.webcam_window:
            self.webcam_window.close()
        self.close()
        cv2.destroyAllWindows()
        QApplication.instance().quit()


def main():
    if len(sys.argv) != 2:
        print("Usage: python main.py <config.json>")
        return

    config_file = sys.argv[1]
    print(config_file)

    # Load config
    try:
        config = load_config(config_file)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return

    fps = 0
    cap = None
    session_csv_path = None
    session_video_path = None

    # Initialize capture
    if config['source'] == 'webcam':
        cap = cv2.VideoCapture(0)
        fps = 30
    elif config['source'] in ['image', 'video']:
        session_video_path = config.get("session_video")
        session_csv_path = config.get("session_csv")
        if session_video_path:
            print(f"Processing session video: {session_video_path}")
            cap = cv2.VideoCapture(session_video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps == 0 or fps is None:
                print("Warning: Unable to determine FPS. Defaulting to 30.")
                fps = 30
    else:
        print("Error: Invalid source type. Use 'webcam', 'image', or 'video'.")
        return

    app = QApplication(sys.argv)
    gaze_window = GazeTrackingApp(config, fps)

    if config['source'] == 'webcam' or session_video_path:
        gaze_window.cap = cap
        gaze_window.resize(1280, 720)
        gaze_window.show()
        if not cap.isOpened():
            print("Error: Unable to open video source.")
            return

    # Initialize HMM
    gaze_window.hmm = HiddenMarkovModel()

    # Handle baseline
    if config.get("baseline", False):
        baseline_video_path = config.get("baseline_video")
        baseline_csv_path = config.get("baseline_csv")

        if baseline_csv_path:
            print(f"Loading baseline CSV from: {baseline_csv_path}")
            # Now load x, y, time from baseline
            x_arr, y_arr, t_arr = load_features_data(baseline_csv_path)
            features = gaze_window.hmm.prepare_features(x_arr, y_arr, t_arr)
            gaze_window.hmm.train(features)
            print("HMM training completed using baseline CSV data.")

        elif baseline_video_path:
            print(f"Processing baseline video: {baseline_video_path}")
            bx, by, bt = process_baseline_video(baseline_video_path)
            # Instead of storing features, store x,y,time
            export_features_csv(bx, by, bt, os.path.join(config["export_dir"], "baseline.csv"))

            # Now reload them from CSV
            x_arr, y_arr, t_arr = load_features_data(os.path.join(config["export_dir"], "baseline.csv"))
            features = gaze_window.hmm.prepare_features(x_arr, y_arr, t_arr)
            gaze_window.hmm.train(features)
            print("HMM training completed using processed baseline video data.")
        else:
            print("Error: Baseline enabled but no path for baseline CSV/video.")
            return

    if session_csv_path:
        # If session CSV is given => load x,y,time
        print(f"Loading session CSV from: {session_csv_path}")
        x_arr, y_arr, t_arr = load_features_data(session_csv_path)
        features = gaze_window.hmm.prepare_features(x_arr, y_arr, t_arr)
        deception_data = gaze_window.hmm.predict_proba(features)[:, 1]
        aligned_time = t_arr[2:]

        if len(aligned_time) != len(deception_data):
            print("Time/deception mismatch after alignment!")
            return

        # Export final if needed
        if config['export']['csv'] or config['export']['graph']:
            if not os.path.exists(config['export_dir']):
                os.makedirs(config['export_dir'], exist_ok=True)

        if config['export']['csv']:
            csv_path = os.path.join(config['export_dir'], "gaze_data.csv")
            export_csv(features, aligned_time, deception_data, csv_path)
            print(f"CSV file saved to: {csv_path}")

        if config['export']['graph']:
            graph_path = os.path.join(config['export_dir'], "final_comprehensive_plots.png")
            export_graph(features, aligned_time, deception_data, graph_path)
        print("Session processing completed using session CSV.")
        return

    elif session_video_path:
        # Real-time approach
        gaze_window.webcam_window = WebcamWindow()
        gaze_window.webcam_window.show()

        def update_frame():
            if not gaze_window.paused:
                ret, frame = gaze_window.cap.read()
                if not ret:
                    gaze_window.cap.release()
                    return
                try:
                    stabilized_frame, iris_detected = process_frame(
                        frame, gaze_window.x_data, gaze_window.y_data,
                        config['affine'], config['dot_display'], config['categorize']
                    )
                    if iris_detected:
                        current_time = gaze_window.frame_count / gaze_window.fps
                        gaze_window.time_data.append(current_time)
                        gaze_window.frame_count += 1

                        gaze_window.x_curve.setData(gaze_window.time_data, gaze_window.x_data)
                        gaze_window.y_curve.setData(gaze_window.time_data, gaze_window.y_data)
                        gaze_window.scatter.setData(gaze_window.x_data, gaze_window.y_data)

                    gaze_window.webcam_window.update_video(stabilized_frame)
                except Exception as e:
                    print(f"Error processing frame: {e}")

        timer = QTimer()
        timer.timeout.connect(update_frame)
        timer.start(30)
        sys.exit(app.exec_())

    else:
        print("Error: No session video or session CSV specified.")
        return


if __name__ == "__main__":
    print("Starting gaze tracking application...")
    main()
