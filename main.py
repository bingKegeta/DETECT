import cv2
import numpy as np
import sys
import os
import pyqtgraph as pg
from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QHBoxLayout, QWidget, QPushButton
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt
from src.args import load_config
from src.utils import load_data
from src.export import export_csv, export_graph, export_features_csv
from src.analysis import Analysis
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
        self.var_data = []
        self.acc_data = []

        # Create a new instance of the Analysis class
        self.analysis = Analysis()

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
            # 1) Export raw x,y,time => session.csv
            session_csv_path = os.path.join(self.config["export_dir"], "session.csv")
            export_features_csv(self.x_data, self.y_data, self.time_data, session_csv_path)

            # 2) If we want final CSV with deception => we line them up
            min_len = min(len(self.time_data), len(self.deception_data), len(self.var_data), len(self.acc_data))

            aligned_time = self.time_data[:min_len]
            aligned_var  = self.var_data[:min_len]
            aligned_acc  = self.acc_data[:min_len]
            aligned_dec  = self.deception_data[:min_len]

            # Build your final "features" => shape (N,3)
            # [time, scaled_variance, scaled_acceleration]
            arr_features = np.column_stack([aligned_time, aligned_var, aligned_acc])

            # Then pass those to export_csv or export_graph
            if self.config['export']['csv']:
                final_csv_path = os.path.join(self.config['export_dir'], "gaze_data.csv")
                export_csv(arr_features, aligned_time, aligned_dec, final_csv_path)

            if self.config['export']['graph']:
                final_graph_path = os.path.join(self.config['export_dir'], "final_comprehensive_plots.png")
                export_graph(arr_features, aligned_time, aligned_dec, final_graph_path)

            print("Session CSV processing completed.")

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

    fps = 30
    cap = None
    session_video_path = config.get("session_video", None)
    session_csv_path   = config.get("session_csv", None)

    # Initialize Qt
    app = QApplication(sys.argv)
    gaze_window = GazeTrackingApp(config, fps)

    if config['source'] == 'webcam':
        cap = cv2.VideoCapture(0)
        fps = 30
        gaze_window.cap = cap
        if not cap.isOpened():
            print("Error: Unable to open webcam.")
            return

    elif config['source'] in ['image', 'video']:
        if session_video_path:
            print(f"Processing session video: {session_video_path}")
            cap = cv2.VideoCapture(session_video_path)
            actual_fps = cap.get(cv2.CAP_PROP_FPS)
            if actual_fps and actual_fps > 0:
                fps = actual_fps
            else:
                print("Warning: Unable to determine FPS. Defaulting to 30.")
                fps = 30
            gaze_window.cap = cap
        elif session_csv_path:
            print(f"Loading session CSV from: {session_csv_path}")

            # 1) Load raw data from CSV
            x_arr, y_arr, t_arr = load_data(session_csv_path)

            # Check we have some data
            if len(t_arr) == 0:
                print("No data found in the session CSV.")
                return

            # 2) Process each point, just like real-time
            for i in range(len(t_arr)):
                # Append to the main window's raw arrays
                gaze_window.x_data.append(x_arr[i])
                gaze_window.y_data.append(y_arr[i])
                gaze_window.time_data.append(t_arr[i])

                # Single-update to get deception probability
                var_scaled, acc_scaled, prob = gaze_window.analysis.single_update(
                    t_arr[i], x_arr[i], y_arr[i]
                )

                gaze_window.var_data.append(var_scaled)
                gaze_window.acc_data.append(acc_scaled)
                gaze_window.deception_data.append(prob)

            min_len = min(len(gaze_window.time_data), len(gaze_window.deception_data), len(gaze_window.var_data), len(gaze_window.acc_data))

            aligned_time = gaze_window.time_data[:min_len]
            aligned_var  = gaze_window.var_data[:min_len]
            aligned_acc  = gaze_window.acc_data[:min_len]
            aligned_dec  = gaze_window.deception_data[:min_len]

            # # Log max and min and mean and medien and std values for each feature and X, Y
            # print(f"Max X: {max(gaze_window.x_data)}, Min X: {min(gaze_window.x_data)}, Mean X: {np.mean(gaze_window.x_data)}, Median X: {np.median(gaze_window.x_data)}, Std X: {np.std(gaze_window.x_data)}")
            # print(f"Max Y: {max(gaze_window.y_data)}, Min Y: {min(gaze_window.y_data)}, Mean Y: {np.mean(gaze_window.y_data)}, Median Y: {np.median(gaze_window.y_data)}, Std Y: {np.std(gaze_window.y_data)}")
            # print(f"Max var: {max(aligned_var)}, Min var: {min(aligned_var)}, Mean var: {np.mean(aligned_var)}, Median var: {np.median(aligned_var)}, Std var: {np.std(aligned_var)}")
            # print(f"Max acc: {max(aligned_acc)}, Min acc: {min(aligned_acc)}, Mean acc: {np.mean(aligned_acc)}, Median acc: {np.median(aligned_acc)}, Std acc: {np.std(aligned_acc)}")

            # Build your final "features" => shape (N,3)
            # [time, scaled_variance, scaled_acceleration]
            arr_features = np.column_stack([aligned_time, aligned_var, aligned_acc])

            # Then pass those to export_csv or export_graph
            if gaze_window.config['export']['csv']:
                final_csv_path = os.path.join(gaze_window.config['export_dir'], "gaze_data.csv")
                export_csv(arr_features, aligned_time, aligned_dec, final_csv_path)

            if gaze_window.config['export']['graph']:
                final_graph_path = os.path.join(gaze_window.config['export_dir'], "final_comprehensive_plots.png")
                export_graph(arr_features, aligned_time, aligned_dec, final_graph_path)

            print("Session CSV processing (point-by-point) completed.")
            return

        else:
            print("Error: No session_video or session_csv provided for 'video' source.")
            return
    else:
        print("Error: Invalid source. Must be 'webcam', 'image', or 'video'.")
        return

    gaze_window.resize(1280, 720)
    gaze_window.show()

    # If we do have a capture device
    if gaze_window.cap:
        gaze_window.webcam_window = WebcamWindow()
        gaze_window.webcam_window.show()

        from src.process import process_frame

        def update_frame():
            if not gaze_window.paused:
                ret, frame = gaze_window.cap.read()
                if not ret:
                    gaze_window.cap.release()
                    return
                try:
                    stabilized_frame, iris_detected = process_frame(
                        frame, 
                        gaze_window.x_data,
                        gaze_window.y_data,
                        config['affine'], 
                        config['dot_display'],
                        config['categorize']
                    )
                    if iris_detected:
                        current_time = gaze_window.frame_count / gaze_window.fps
                        gaze_window.time_data.append(current_time)
                        gaze_window.frame_count += 1

                        # Real-time plot updates
                        gaze_window.x_curve.setData(gaze_window.time_data, gaze_window.x_data)
                        gaze_window.y_curve.setData(gaze_window.time_data, gaze_window.y_data)
                        gaze_window.scatter.setData(gaze_window.x_data, gaze_window.y_data)

                    # Now get deception probability per frame
                    # last x,y => gaze_window.x_data[-1], gaze_window.y_data[-1]
                    var_scaled, acc_scaled, prob = gaze_window.analysis.single_update(
                        current_time, 
                        gaze_window.x_data[-1], 
                        gaze_window.y_data[-1]
                    )
                    gaze_window.var_data.append(var_scaled)
                    gaze_window.acc_data.append(acc_scaled)
                    gaze_window.deception_data.append(prob)

                    gaze_window.webcam_window.update_video(stabilized_frame)
                except Exception as e:
                    print(f"Error processing frame: {e}")

        timer = QTimer()
        timer.timeout.connect(update_frame)
        timer.start(30)

        sys.exit(app.exec_())
    else:
        # No real-time approach, might just rely on close_app or session CSV
        sys.exit(app.exec_())

if __name__ == "__main__":
    print("Starting gaze tracking (one point at a time) application...")
    main()