import cv2
import time
import numpy as np
import sys
import os
import pyqtgraph as pg
from src.args import load_config
from src.process import process_frame
from src.animation import create_gaze_animation
from src.graph import plot_gaze_tracking, plot_final_graphs
from src.export import export_csv

def main():
    # Ensure that the JSON file path is provided as a command-line argument
    if len(sys.argv) != 2:
        print("Usage: python main.py <config.json>")
        return
    
    config_file = sys.argv[1]

    # Load the configuration from the JSON file
    try:
        config = load_config(config_file)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return

    frame_count = 0  # Initialize frame counter for FPS calculation
    fps = None  # Initialize FPS variable

    # Initialize video capture based on the configuration
    if config['source'] == 'webcam':
        cap = cv2.VideoCapture(0)
        start_time = time.time()  # Start time for FPS calculation
    elif config['source'] in ['image', 'video']:
        if not config['path']:
            print("Error: 'path' field is required in the configuration when 'source' is 'image' or 'video'.")
            return
        cap = cv2.VideoCapture(config['path'])
        fps = cap.get(cv2.CAP_PROP_FPS)  # Get FPS for video files
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

    # Get video properties
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to read the first frame.")
        cap.release()
        return

    height, width = frame.shape[:2]

    # Reinitialize video capture to start from the first frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Initialize data arrays to store gaze coordinates over time
    x_data = []
    y_data = []
    time_data = []

    # Initialize PyQt Application
    app = pg.mkQApp("Gaze Tracking")  # Create the PyQt application
    pg.setConfigOptions(antialias=True)

    # Set up real-time graphing if required
    if config['graph']:
        win = pg.GraphicsLayoutWidget(show=True, title="Gaze Tracking")
        win.resize(800, 600)

        # Line Plot for X Coordinate
        ax_x = win.addPlot(title='X Coordinate Over Time')
        ax_x.setLabel('left', 'X Coordinate')
        ax_x.setLabel('bottom', 'Time (s)')
        x_curve = ax_x.plot(pen='r')

        # Line Plot for Y Coordinate
        ax_y = win.addPlot(title='Y Coordinate Over Time', row=1, col=0)
        ax_y.setLabel('left', 'Y Coordinate')
        ax_y.setLabel('bottom', 'Time (s)')
        y_curve = ax_y.plot(pen='b')

        # 2D Scatter Plot
        scatter_ax = win.addPlot(title='2D Gaze Points Over Time', row=2, col=0)
        scatter = pg.ScatterPlotItem(pen='g')
        scatter_ax.addItem(scatter)

        # Create a separate window for the 3D scatter plot
        scatter3d_win = pg.GraphicsLayoutWidget(show=True, title="3D Gaze Points")
        scatter3d_ax = scatter3d_win.addPlot(title='3D Gaze Points')
        scatter3d_ax.setLabel('left', 'Vertical Coordinate')
        scatter3d_ax.setLabel('bottom', 'Horizontal Coordinate')
        scatter3d_ax.setLabel('right', 'Time (s)')
        scatter3d = pg.ScatterPlotItem(size=5, pen='y')  # Create a scatter plot item for 3D
        scatter3d_ax.addItem(scatter3d)

    last_csv_time = 0  # For controlling CSV export interval

    # Ensure the export directory exists or create it
    if config['export']['csv'] or config['export']['graph']:
        try:
            os.makedirs(config['export_dir'], exist_ok=True)
        except Exception as e:
            print(f"Error creating export directory '{config['export_dir']}': {e}")
            return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if fps is None:
            current_time = time.time() - start_time  # For webcam: Use real-time timestamp
        else:
            frame_count += 1  # Increment frame counter
            current_time = frame_count / fps  # For video files: Use FPS-based timestamp

        # Process the frame and update gaze data
        stabilized_frame, iris_detected = process_frame(frame, x_data, y_data, config['affine'], config['dot_display'], config['categorize'])

        # If gaze points were detected, update time and plot
        if iris_detected:
            time_data.append(current_time)

            if config['graph']:
                # Update plots
                x_curve.setData(time_data, x_data)
                y_curve.setData(time_data, y_data)

                # Update 2D scatter plot
                scatter.setData(x_data, y_data)

                # Update 3D scatter plot
                scatter3d.setData(x_data, y_data, pen='y')  # This is still incorrect for 3D

            # Export data to CSV at specified interval, if enabled
            if config['export']['csv'] and current_time - last_csv_time >= config['csv_interval']:
                csv_path = os.path.join(config['export_dir'], "raw_data.csv")
                export_csv(x_data, y_data, time_data, csv_path)
                last_csv_time = current_time

        # Display video with iris tracking
        cv2.imshow('Gaze Tracking', stabilized_frame)

        # Exit on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up
    cap.release()
    cv2.destroyAllWindows()

    gaze_positions = np.column_stack((x_data, y_data)).tolist()  # Convert to list of (x, y) tuples
    # Calculate total duration
    if config['source'] == 'webcam':
        if time_data and time_data[-1] > 0:
            fps = len(gaze_positions) / time_data[-1]
        else:
            fps = 30  # Default FPS if unable to calculate
    
    # Create gaze animation if animation export is enabled
    if config['export']['animation']:
        create_gaze_animation(gaze_positions, width, height, fps=int(fps), save_path=config['animation_out'])

    # Create the graph image if graph export is enabled
    if config['export']['graph']:
        plot_final_graphs(time_data, x_data, y_data, save_path=config['graph_out'])

if __name__ == "__main__":
    main()
