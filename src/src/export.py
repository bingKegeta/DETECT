import csv
import pyqtgraph as pg
import os

def export_graph(time_data, x_data, y_data, deception_data, save_path):
    """
    Export gaze tracking graphs (X, Y coordinates, deception probabilities, and scatter plot) to an image file.
    :param time_data: List of time values.
    :param x_data: List of X coordinates.
    :param y_data: List of Y coordinates.
    :param deception_data: List of deception probabilities.
    :param save_path: Path to save the graph image.
    """
    try:
        # Ensure export directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Create an exportable plot widget
        export_widget = pg.GraphicsLayoutWidget(show=False)
        export_widget.resize(800, 600)

        # X Coordinate plot
        ax_x = export_widget.addPlot(title='X Coordinate Over Time')
        ax_x.setLabel('left', 'X Coordinate')
        ax_x.setLabel('bottom', 'Time (s)')
        ax_x.plot(time_data, x_data, pen='r')

        # Y Coordinate plot
        ax_y = export_widget.addPlot(title='Y Coordinate Over Time', row=1, col=0)
        ax_y.setLabel('left', 'Y Coordinate')
        ax_y.setLabel('bottom', 'Time (s)')
        ax_y.plot(time_data, y_data, pen='b')

        # Deception Probability plot
        ax_prob = export_widget.addPlot(title='Deception Probability Over Time', row=2, col=0)
        ax_prob.setLabel('left', 'Probability')
        ax_prob.setLabel('bottom', 'Time (s)')
        ax_prob.plot(time_data, deception_data, pen='g')

        # Scatter plot for 2D gaze points
        scatter_ax = export_widget.addPlot(title='2D Gaze Points', row=3, col=0)
        scatter = pg.ScatterPlotItem(x=x_data, y=y_data, pen=None, brush=pg.mkBrush(0, 255, 0, 120))
        scatter_ax.addItem(scatter)

        # Export the graph as an image
        screenshot = export_widget.grab()
        screenshot.save(save_path, 'PNG')
        print(f"Graph image saved to: {save_path}")
    except Exception as e:
        print(f"Error exporting graph: {e}")

def export_csv(time_data, x_data, y_data, features, deception_data, output_file):
    """
    Export gaze data and corresponding deception probabilities to CSV.
    :param time_data: List of time values
    :param x_data: List of x gaze positions
    :param y_data: List of y gaze positions
    :param features: Normalized features (variance, velocity, acceleration)
    :param deception_data: Deception probabilities
    :param output_file: Output CSV file path
    """
    if len(time_data) == 0 or len(features) == 0:
        print("No data to export.")
        return

    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Time', 'X Coordinate', 'Y Coordinate', 'Variance', 'Velocity', 'Acceleration', 'Deception Probability'])

        for i in range(len(time_data)):
            variance = features[i, 1] if i < len(features) else 0
            velocity = features[i, 2] if i < len(features) else 0
            acceleration = features[i, 3] if i < len(features) else 0
            deception = deception_data[i] if i < len(deception_data) else None

            writer.writerow([time_data[i], x_data[i], y_data[i], variance, velocity, acceleration, deception])
