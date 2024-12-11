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
from src.export import export_csv
from src.graph import plot_final_graphs    
from src.utils import load_baseline_data
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
        self.video_label.setScaledContents(True)  # Allows the video feed to scale
        self.video_label.setFixedSize(640, 480)  # Set webcam display size

        layout = QVBoxLayout()
        layout.addWidget(self.video_label)
        self.setLayout(layout)

    def update_video(self, frame):
        """Convert the frame to QImage and update the video QLabel."""
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        q_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(q_image))  # Update QLabel with QImage


class GazeTrackingApp(QWidget):
    def __init__(self, config, fps=30):
        super().__init__()
        self.config = config
        self.initUI()
        self.paused = False  # Initialize pause state
        self.cap = None  # Video capture object will be initialized in main
        self.webcam_window = None  # To hold the webcam window reference

        # Initialize data arrays to store gaze coordinates over time
        self.fps = fps  # Frames per second for precise timestamp calculations
        self.frame_count = 0  # Frame counter to calculate precise timestamps
        self.time_data = []
        self.x_data = []
        self.y_data = []
        self.deception_data = []  # Initialize deception data to store probabilities

    def initUI(self):
        # Main layout
        main_layout = QVBoxLayout(self)
        self.setWindowTitle("Graphs")

        # Set up real-time graphing for gaze data
        self.win = pg.GraphicsLayoutWidget(show=True, title="Gaze Tracking")
        main_layout.addWidget(self.win, stretch=1)  # Add graphs to the main layout

        # Add line plots and scatter plot to the graphics view
        self.ax_x = self.win.addPlot(title='X Coordinate Over Time')
        self.ax_x.setLabel('left', 'X Coordinate')
        self.ax_x.setLabel('bottom', 'Time (s)')
        self.x_curve = self.ax_x.plot(pen='r')  # X coordinate curve

        self.ax_y = self.win.addPlot(title='Y Coordinate Over Time', row=1, col=0)
        self.ax_y.setLabel('left', 'Y Coordinate')
        self.ax_y.setLabel('bottom', 'Time (s)')
        self.y_curve = self.ax_y.plot(pen='b')  # Y coordinate curve

        self.ax_prob = self.win.addPlot(title='Deception Probability Over Time', row=3, col=0)
        self.ax_prob.setLabel('left', 'Probability')
        self.ax_prob.setLabel('bottom', 'Time (s)')
        self.prob_curve = self.ax_prob.plot(pen='g')  # Deception probability curve

        self.scatter_ax = self.win.addPlot(title='2D Gaze Points Over Time', row=2, col=0)
        self.scatter = pg.ScatterPlotItem(pen='g')  # Scatter plot for 2D points
        self.scatter_ax.addItem(self.scatter)

        # Play/Pause Button
        self.play_pause_button = QPushButton("Pause")
        self.play_pause_button.clicked.connect(self.toggle_pause)
        self.play_pause_button.setStyleSheet("background-color: lightblue; color: white;")  # Set button color to blue
        main_layout.addWidget(self.play_pause_button)

        # Close Button
        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.close_app)
        self.close_button.setStyleSheet("background-color: red; color: white;")  # Set button color to red
        main_layout.addWidget(self.close_button)

    def toggle_pause(self):
        self.paused = not self.paused
        self.play_pause_button.setText("Play" if self.paused else "Pause")

    def close_app(self):
        """Stop the webcam, export data, and close both windows."""
        # Export CSV and graph before closing the application
        if self.config['export']['csv'] or self.config['export']['graph']:
            if not os.path.exists(self.config['export_dir']):
                os.makedirs(self.config['export_dir'], exist_ok=True)

        if self.config['export']['csv']:
            csv_path = os.path.join(self.config['export_dir'], "gaze_data.csv")
            export_csv(self.x_data, self.y_data, self.time_data, self.deception_data, csv_path)
            print(f"CSV file saved to: {csv_path}")

        if self.config['export']['graph']:
            graph_path = os.path.join(self.config['export_dir'], "final_comprehensive_plots.png")
            self.export_graph(graph_path)

        # Close all windows and terminate the application properly
        self.cleanup()

    def export_graph(self, save_path):
        """Function to export the graph to a file."""
        # Create a new plot instance to export the data
        app = QApplication(sys.argv)  # Ensure we have a running QApplication

        plot_widget = pg.GraphicsLayoutWidget(show=False)
        plot_widget.resize(800, 600)

        ax_x = plot_widget.addPlot(title='X Coordinate Over Time')
        ax_x.setLabel('left', 'X Coordinate')
        ax_x.setLabel('bottom', 'Time (s)')
        ax_x.plot(self.time_data, self.x_data, pen='r')

        ax_y = plot_widget.addPlot(title='Y Coordinate Over Time', row=1, col=0)
        ax_y.setLabel('left', 'Y Coordinate')
        ax_y.setLabel('bottom', 'Time (s)')
        ax_y.plot(self.time_data, self.y_data, pen='b')

        scatter_ax = plot_widget.addPlot(title='2D Gaze Points Over Time', row=2, col=0)
        scatter = pg.ScatterPlotItem(x=self.x_data, y=self.y_data, pen='g')
        scatter_ax.addItem(scatter)

        ax_prob = plot_widget.addPlot(title='Deception Probability Over Time', row=3, col=0)
        ax_prob.setLabel('left', 'Probability')
        ax_prob.setLabel('bottom', 'Time (s)')
        ax_prob.plot(self.time_data, self.deception_data, pen='g')

        # Take a screenshot of the plot widget and save it
        screenshot = plot_widget.grab()
        screenshot.save(save_path, 'PNG')
        print(f"Graph image saved to: {save_path}")

        app.quit()  # Properly quit the QApplication after saving

    def cleanup(self):
        """Release the webcam and close all windows."""
        if self.cap:
            self.cap.release()  # Release the webcam
        if self.webcam_window:
            self.webcam_window.close()  # Close the webcam window
        self.close()  # Close the gaze tracking window
        cv2.destroyAllWindows()  # Close any OpenCV windows if they exist
        QApplication.instance().quit()  # Quit the application
        
def main():
    # Ensure that the JSON file path is provided as a command-line argument
    if len(sys.argv) != 2:
        print("Usage: python main.py <config.json>")
        return
    
    config_file = sys.argv[1]

    print(config_file)

    # Load the configuration from the JSON file
    try:
        config = load_config(config_file)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return
    
    # Initialize video capture based on the configuration
    if config['source'] == 'webcam':
        cap = cv2.VideoCapture(0)
        fps = 30  # Assume default FPS for webcam input
    elif config['source'] in ['image', 'video']:
        if not config['path']:
            print("Error: 'path' field is required in the configuration when 'source' is 'image' or 'video'.")
            return
        cap = cv2.VideoCapture(config['path'])
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0 or fps is None:
            print("Warning: Unable to determine FPS from video. Defaulting to 30.")
            fps = 30  # Default FPS if unable to get from video source
    else:
        print("Error: Invalid source type provided in the configuration. Use 'webcam', 'image', or 'video'.")
        return

    # Check if video capture is initialized
    if not cap.isOpened():
        print("Error: Unable to open video source. Check the camera index or file path.")
        return

    # Initialize the QApplication
    app = QApplication(sys.argv)

    # Create main window for gaze tracking
    gaze_window = GazeTrackingApp(config, fps)
    gaze_window.cap = cap
    gaze_window.resize(1280, 720)  # Set a reasonable starting size for the gaze window
    gaze_window.show()

    # Initialize the HMM for deception detection
    gaze_window.hmm = HiddenMarkovModel()

    # Handle baseline setup and train the HMM
    if config.get("baseline", False):
        baseline_path = config.get("baseline_path", None)
        baseline_csv_path = config.get("baseline_csv", None)

        if baseline_csv_path:
            print(f"Loading baseline CSV from: {baseline_csv_path}")
            baseline_x, baseline_y, baseline_time = load_baseline_data(baseline_csv_path)
            print(f"Loaded {len(baseline_x)} data points from baseline CSV.")
            gaze_window.hmm.train(baseline_x, baseline_y, baseline_time)
            print("HMM training completed using baseline CSV data.")
        elif baseline_path:
            print(f"Processing baseline video: {baseline_path}")
            baseline_data = process_baseline_video(baseline_path)
            export_csv(*baseline_data, os.path.join(config["export_dir"], "baseline.csv"))
            baseline_x, baseline_y, baseline_time = baseline_data
            gaze_window.hmm.train(baseline_x, baseline_y, baseline_time)
            print("HMM training completed using processed baseline video data.")
        else:
            print("Error: Baseline enabled but no path provided for baseline CSV or video.")
            return

    # Create a separate window for the webcam feed
    gaze_window.webcam_window = WebcamWindow()  # Store the webcam window in the gaze_window
    gaze_window.webcam_window.show()

    def update_frame():
        if not gaze_window.paused:
            ret, frame = gaze_window.cap.read()
            if not ret:
                gaze_window.cap.release()
                return

            # Process the frame and update gaze data
            try:
                stabilized_frame, iris_detected = process_frame(
                    frame, gaze_window.x_data, gaze_window.y_data,
                    config['affine'], config['dot_display'], config['categorize']
                )
                
                # Only update plots if eyes are detected
                if iris_detected:
                    # Use FPS and frame count to calculate timestamp
                    current_time = gaze_window.frame_count / gaze_window.fps
                    gaze_window.time_data.append(current_time)
                    gaze_window.frame_count += 1  # Increment frame counter

                    # Use the HMM to calculate deception probability
                    prob = gaze_window.hmm.predict_proba(gaze_window.x_data[-1], gaze_window.y_data[-1], current_time)
                    gaze_window.deception_data.append(prob[0][1])  # Append probability of "deception" state

                    # Update plots with new gaze data
                    gaze_window.x_curve.setData(gaze_window.time_data, gaze_window.x_data)
                    gaze_window.y_curve.setData(gaze_window.time_data, gaze_window.y_data)
                    gaze_window.scatter.setData(gaze_window.x_data, gaze_window.y_data)

                    # Update the deception probability graph
                    if len(gaze_window.time_data) == len(gaze_window.deception_data):  # Ensure alignment
                        gaze_window.prob_curve.setData(gaze_window.time_data, gaze_window.deception_data)

                # Display the processed frame (regardless of iris detection)
                gaze_window.webcam_window.update_video(stabilized_frame)
            
            except Exception as e:
                print(f"Error processing frame: {e}")

    # Set up a timer to call update_frame at regular intervals
    timer = QTimer()
    timer.timeout.connect(update_frame)
    timer.start(30)  # Adjust interval to control the update frequency

    sys.exit(app.exec_())

if __name__ == "__main__":
    print("Starting gaze tracking application...")
    main()
