# src/animation.py

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

def create_gaze_animation(eye_positions, frame_width, frame_height, fps=30):
    """
    Creates and saves a gaze animation based on eye_positions.

    Args:
        eye_positions (list of tuples): List containing (x, y) gaze coordinates.
        frame_width (int): Width of the video frame.
        frame_height (int): Height of the video frame.
        fps (int): Frames per second for the animation.
    """
    if not eye_positions:
        print("No gaze data available to create animation.")
        return

    # Extract x and y coordinates
    x_values = [x for x, y in eye_positions]
    y_values = [y for x, y in eye_positions]

    # Compute min and max for x and y
    min_x = min(x_values)
    max_x = max(x_values)
    min_y = min(y_values)
    max_y = max(y_values)

    # Calculate ranges
    range_x = max_x - min_x
    range_y = max_y - min_y

    # Add padding (10% of the range)
    padding_x = 0.1 * range_x if range_x != 0 else 1
    padding_y = 0.1 * range_y if range_y != 0 else 1

    # Set plot limits with padding
    ax_min_x = min_x - padding_x
    ax_max_x = max_x + padding_x
    ax_min_y = min_y - padding_y
    ax_max_y = max_y + padding_y

    # Compute center of the gaze data
    center_x = (ax_min_x + ax_max_x) / 2
    center_y = (ax_min_y + ax_max_y) / 2

    # Determine the radius for the artificial eye
    radius = max(range_x, range_y) / 2 + max(padding_x, padding_y)

    # Initialize the plot
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(ax_min_x, ax_max_x)
    ax.set_ylim(ax_min_y, ax_max_y)
    ax.set_xlabel('Horizontal Gaze')
    ax.set_ylabel('Vertical Gaze')
    ax.set_title('Gaze Animation')
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True)

    # Draw the artificial eye centered at (center_x, center_y)
    eye = plt.Circle((center_x, center_y), radius, edgecolor='black', facecolor='white', linewidth=2)
    ax.add_patch(eye)

    # Initialize the gaze point
    point, = ax.plot([], [], 'o', color='blue', markersize=15)

    def init():
        point.set_data([], [])
        return point,

    def animate_frame(i):
        if i < len(eye_positions):
            x, y = eye_positions[i]
            if x is not None and y is not None:
                # Directly set the gaze point without additional normalization
                point.set_data([x], [y])
            else:
                # If gaze is not detected, center the gaze point
                point.set_data([center_x], [center_y])
        return point,

    ani = animation.FuncAnimation(
        fig,
        animate_frame,
        init_func=init,
        frames=len(eye_positions),
        interval=1000 / fps,  # interval in milliseconds
        blit=True,
        repeat=False
    )

    try:
        # Save the animation as a video file
        ani.save('animation.mp4', writer='ffmpeg', fps=fps)
        print("Gaze animation saved as 'animation.mp4'")
    except FileNotFoundError:
        print("Error: ffmpeg is not installed or not found in PATH.")
    except Exception as e:
        print(f"Failed to save animation: {e}")

    plt.close(fig)