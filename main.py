import cv2
import time
import matplotlib.pyplot as plt
from src.process import process_frame
from src.animation import create_gaze_animation
from src.graph import plot_gaze_tracking, plot_final_graphs
from src.export import export_csv
from src.args import parse_args
import numpy as np

def main():
    args = parse_args()

    # Initialize video capture based on CLI input
    if args.source == 'webcam':
        cap = cv2.VideoCapture(1)
    elif args.source in ['image', 'video']:
        if not args.path:
            print("Error: --path argument is required when --source is 'image' or 'video'.")
            return
        cap = cv2.VideoCapture(args.path)
    else:
        print("Error: Invalid source type provided. Use --source with 'webcam', 'image', or 'video'.")
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
    heatmap_data = np.zeros((height, width), dtype=np.float32)

    # Reinitialize video capture to start from the first frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Initialize data arrays to store gaze coordinates over time
    x_data = []
    y_data = []
    time_data = []

    # Set up real-time graphing if required
    if args.graph:
        plt.ion()

        # Create a figure with 4 subplots: 2 line plots, 1 2D scatter, and 1 3D scatter
        fig = plt.figure(figsize=(15, 10))

        # Line Plot for X Coordinate
        ax_x = fig.add_subplot(2, 2, 1)
        ax_x.set_title('X Coordinate Over Time')
        ax_x.set_xlabel('Time (s)')
        ax_x.set_ylabel('X Coordinate')
        ax_x.legend(['X Coordinate (Left-Right)'])
        ax_x.grid(True)

        # Line Plot for Y Coordinate
        ax_y = fig.add_subplot(2, 2, 2)
        ax_y.set_title('Y Coordinate Over Time')
        ax_y.set_xlabel('Time (s)')
        ax_y.set_ylabel('Y Coordinate')
        ax_y.legend(['Y Coordinate (Up-Down)'])
        ax_y.grid(True)

        # 2D Scatter Plot
        scatter_ax = fig.add_subplot(2, 2, 3)
        scatter_ax.set_title('2D Gaze Points Over Time')
        scatter_ax.set_xlabel('Horizontal Coordinate')
        scatter_ax.set_ylabel('Vertical Coordinate')
        scatter_ax.invert_yaxis()

        # 3D Scatter Plot
        scatter3d_ax = fig.add_subplot(2, 2, 4, projection='3d')
        scatter3d_ax.set_title('3D Gaze Points')
        scatter3d_ax.set_xlabel('Horizontal Coordinate')
        scatter3d_ax.set_ylabel('Vertical Coordinate')
        scatter3d_ax.set_zlabel('Time (s)')
        scatter3d_ax.invert_yaxis()

    start_time = time.time()
    last_csv_time = 0  # For controlling CSV export interval

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        current_time = time.time() - start_time

        # Process the frame and update gaze data
        stabilized_frame, iris_detected = process_frame(frame, x_data, y_data, args.affine)

        # If gaze points were detected, update time and plot
        if iris_detected:
            time_data.append(current_time)

            if args.graph:
                plot_gaze_tracking(time_data, x_data, y_data, ax_x, ax_y, scatter_ax, scatter3d_ax)

            # # Update heatmap data
            # # Assuming x_data and y_data are normalized between -1 and 1
            # # Convert normalized coordinates to pixel indices
            # x_pixel = int((x_data[-1] + 1) * (width / 2))
            # y_pixel = int((y_data[-1] + 1) * (height / 2))
            # if 0 <= x_pixel < width and 0 <= y_pixel < height:
            #     heatmap_data[y_pixel, x_pixel] += 1

            # Export data to CSV at specified interval
            if args.csv and current_time - last_csv_time >= args.csv_interval:
                export_csv(x_data, y_data, time_data, args.csv)
                last_csv_time = current_time

        # Display video with iris tracking
        cv2.imshow('Gaze Tracking', stabilized_frame)

        # Exit on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up
    cap.release()
    cv2.destroyAllWindows()

    if args.graph:
        gaze_positions = np.column_stack((x_data, y_data)).tolist() # Convert to list of (x, y) tuples
        
        # Calculate total duration
        if time_data:
            total_duration = time_data[-1] - time_data[0]
            if total_duration > 0:
                fps = len(gaze_positions) / total_duration
            else:
                fps = 30  # Default FPS if duration is zero
        else:
            fps = 30

        create_gaze_animation(gaze_positions, width, height, fps=int(fps)) # Create gaze animation

        plot_final_graphs(time_data, x_data, y_data, heatmap_data) # Plot final graphs

if __name__ == "__main__":
    main()