import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

WINDOW_LENGTH = 13
POLYORDER = 2

def plot_gaze_tracking(time_data, x_data, y_data, ax_x, ax_y):
    if len(x_data) == 0 or len(y_data) == 0 or len(time_data) == 0:
        return  # Skip plotting if there's no valid data

    x_data_smoothed = savgol_filter(x_data, WINDOW_LENGTH, POLYORDER) if len(x_data) >= WINDOW_LENGTH else x_data
    y_data_smoothed = savgol_filter(y_data, WINDOW_LENGTH, POLYORDER) if len(y_data) >= WINDOW_LENGTH else y_data

    ax_x.clear()
    ax_y.clear()

    ax_x.plot(time_data, x_data_smoothed, label="X Coordinate (Left-Right)", color='r')
    ax_x.set_xlabel('Time (s)')
    ax_x.set_ylabel('X Coordinate')
    ax_x.legend()

    ax_y.plot(time_data, y_data_smoothed, label="Y Coordinate (Up-Down)", color='b')
    ax_y.set_xlabel('Time (s)')
    ax_y.set_ylabel('Y Coordinate')
    ax_y.legend()

    plt.pause(0.01)
