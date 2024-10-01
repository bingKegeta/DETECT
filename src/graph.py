# src/graph.py

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import savgol_filter
import numpy as np

WINDOW_LENGTH = 13
POLYORDER = 2

def plot_gaze_tracking(time_data, x_data, y_data, ax_x, ax_y, scatter_ax, scatter3d_ax):
    """
    Update real-time plots with the latest gaze data.
    """
    if len(x_data) == 0 or len(y_data) == 0 or len(time_data) == 0:
        return  # Skip plotting if there's no valid data

    # Apply smoothing
    x_data_smoothed = savgol_filter(x_data, WINDOW_LENGTH, POLYORDER) if len(x_data) >= WINDOW_LENGTH else x_data
    y_data_smoothed = savgol_filter(y_data, WINDOW_LENGTH, POLYORDER) if len(y_data) >= WINDOW_LENGTH else y_data

    # Update Line Graphs
    ax_x.clear()
    ax_x.plot(time_data, x_data_smoothed, label="X Coordinate (Left-Right)", color='r')
    ax_x.set_xlabel('Time (s)')
    ax_x.set_ylabel('X Coordinate')
    ax_x.set_title('X Coordinate Over Time')
    ax_x.legend()
    ax_x.grid(True)

    ax_y.clear()
    ax_y.plot(time_data, y_data_smoothed, label="Y Coordinate (Up-Down)", color='b')
    ax_y.set_xlabel('Time (s)')
    ax_y.set_ylabel('Y Coordinate')
    ax_y.set_title('Y Coordinate Over Time')
    ax_y.legend()
    ax_y.grid(True)

    # Update 2D Scatter Plot
    scatter_ax.clear()
    scatter = scatter_ax.scatter(x_data, y_data, c=time_data, cmap='viridis', s=5)
    scatter_ax.set_title('2D Gaze Points Over Time')
    scatter_ax.set_xlabel('Horizontal Coordinate')
    scatter_ax.set_ylabel('Vertical Coordinate')
    scatter_ax.invert_yaxis()
    scatter_ax.legend(['Gaze Points'])

    # Manage colorbar: Add only once
    if not hasattr(plot_gaze_tracking, "cbar_scatter"):
        plot_gaze_tracking.cbar_scatter = scatter_ax.figure.colorbar(scatter, ax=scatter_ax, label='Time (s)')
    else:
        # Update color limits and colorbar
        scatter.set_clim(vmin=min(time_data), vmax=max(time_data))
        plot_gaze_tracking.cbar_scatter.update_normal(scatter)

    # Update 3D Scatter Plot
    scatter3d_ax.clear()
    scatter3d = scatter3d_ax.scatter(x_data, y_data, time_data, c=time_data, cmap='viridis', s=5)
    scatter3d_ax.set_title('3D Gaze Points')
    scatter3d_ax.set_xlabel('Horizontal Coordinate')
    scatter3d_ax.set_ylabel('Vertical Coordinate')
    scatter3d_ax.set_zlabel('Time (s)')
    scatter3d_ax.invert_yaxis()
    scatter3d_ax.legend(['Gaze Points'])

    # Manage colorbar: Add only once
    if not hasattr(plot_gaze_tracking, "cbar_scatter3d"):
        plot_gaze_tracking.cbar_scatter3d = scatter3d_ax.figure.colorbar(scatter3d, ax=scatter3d_ax, label='Time (s)')
    else:
        # Update color limits and colorbar
        scatter3d.set_clim(vmin=min(time_data), vmax=max(time_data))
        plot_gaze_tracking.cbar_scatter3d.update_normal(scatter3d)

    plt.pause(0.001)  # Minimal pause for responsiveness

def plot_final_graphs(time_data, x_data, y_data, heatmap_data):
    """
    Plot all final comprehensive graphs after the session ends.
    """
    plt.ioff()  # Turn off interactive mode

    # Apply smoothing for the final plots
    x_data_smoothed = savgol_filter(x_data, WINDOW_LENGTH, POLYORDER) if len(x_data) >= WINDOW_LENGTH else x_data
    y_data_smoothed = savgol_filter(y_data, WINDOW_LENGTH, POLYORDER) if len(y_data) >= WINDOW_LENGTH else y_data

    # Create a new figure for final plots
    fig = plt.figure(figsize=(20, 15))

    # 1. Line Plot for X Coordinate
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(time_data, x_data_smoothed, label="X Coordinate (Left-Right)", color='r')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('X Coordinate')
    ax1.set_title('Final X Coordinate Over Time')
    ax1.legend()
    ax1.grid(True)

    # 2. Line Plot for Y Coordinate
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(time_data, y_data_smoothed, label="Y Coordinate (Up-Down)", color='b')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Y Coordinate')
    ax2.set_title('Final Y Coordinate Over Time')
    ax2.legend()
    ax2.grid(True)

    # 3. 2D Scatter Plot
    ax3 = fig.add_subplot(2, 2, 3)
    scatter = ax3.scatter(x_data, y_data, c=time_data, cmap='viridis', s=5)
    ax3.set_title('Final 2D Gaze Points')
    ax3.set_xlabel('Horizontal Coordinate')
    ax3.set_ylabel('Vertical Coordinate')
    ax3.invert_yaxis()
    ax3.legend(['Gaze Points'])
    cbar = fig.colorbar(scatter, ax=ax3, label='Time (s)')

    # 4. 3D Scatter Plot
    ax4 = fig.add_subplot(2, 2, 4, projection='3d')
    scatter3d = ax4.scatter(x_data, y_data, time_data, c=time_data, cmap='viridis', s=5)
    ax4.set_title('Final 3D Gaze Points')
    ax4.set_xlabel('Horizontal Coordinate')
    ax4.set_ylabel('Vertical Coordinate')
    ax4.set_zlabel('Time (s)')
    ax4.invert_yaxis()
    ax4.legend(['Gaze Points'])
    cbar3d = fig.colorbar(scatter3d, ax=ax4, label='Time (s)')

    plt.tight_layout()
    plt.savefig('final_comprehensive_plots.png')  # Save all plots as a single image
    plt.show()

    # # Plot Final Heatmap in a Separate Figure
    # plt.figure(figsize=(10, 6))
    # plt.imshow(heatmap_data, cmap='hot', interpolation='nearest', origin='upper')
    # plt.colorbar(label='Gaze Duration')
    # plt.title('Final Gaze Heatmap')
    # plt.xlabel('Horizontal Coordinate')
    # plt.ylabel('Vertical Coordinate')
    # plt.gca().invert_yaxis()
    # plt.savefig('final_gaze_heatmap.png')  # Save heatmap separately
    # plt.show()