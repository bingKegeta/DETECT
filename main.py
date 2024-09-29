import cv2
import time
import matplotlib.pyplot as plt
from src.process import process_frame
from src.graph import plot_gaze_tracking
from src.export import export_csv
from src.args import parse_args

def main():
    args = parse_args()

    # Initialize video capture based on CLI input
    if args.source == 'webcam':
        cap = cv2.VideoCapture(0)
    elif args.source in ['image', 'video']:
        if not args.path:
            print("You must provide a --path to the image or video file.")
            return
        cap = cv2.VideoCapture(args.path)
    else:
        print("Invalid source type provided. Use --source with 'webcam', 'image', or 'video'.")
        return

    # Initialize data arrays to store gaze coordinates over time
    x_data = []
    y_data = []
    time_data = []

    # Set up graphing if required
    if args.graph:
        plt.ion()
        fig, (ax_x, ax_y) = plt.subplots(2, 1)
        ax_x.set_title('X Coordinate vs Time')
        ax_y.set_title('Y Coordinate vs Time')

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
                plot_gaze_tracking(time_data, x_data, y_data, ax_x, ax_y)

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
        plt.close()

if __name__ == "__main__":
    main()
