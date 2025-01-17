import csv
import pyqtgraph as pg
import os

def export_graph(features, time_data, deception_data, save_path):
    """
    Export feature graphs (variance, velocity, acceleration, and deception probabilities) to an image file.
    :param features: Array of features [time, variance, velocity, acceleration].
    :param time_data: List of time values.
    :param deception_data: List of deception probabilities.
    :param save_path: Path to save the graph image.
    """
    try:
        # Ensure export directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Create an exportable plot widget
        export_widget = pg.GraphicsLayoutWidget(show=False)
        export_widget.resize(800, 600)

        # Extract features
        time = features[:, 0]  # Time elapsed
        variance = features[:, 1]  # Normalized variance
        velocity = features[:, 2]  # Normalized velocity
        acceleration = features[:, 3]  # Normalized acceleration

        # Variance plot
        ax_variance = export_widget.addPlot(title="Variance Over Time")
        ax_variance.setLabel('left', 'Variance')
        ax_variance.setLabel('bottom', 'Time (s)')
        ax_variance.plot(time, variance, pen='r')

        # Velocity plot
        ax_velocity = export_widget.addPlot(title="Velocity Over Time", row=1, col=0)
        ax_velocity.setLabel('left', 'Normalized Velocity')
        ax_velocity.setLabel('bottom', 'Time (s)')
        ax_velocity.plot(time, velocity, pen='b')

        # Acceleration plot
        ax_acceleration = export_widget.addPlot(title="Acceleration Over Time", row=2, col=0)
        ax_acceleration.setLabel('left', 'Acceleration')
        ax_acceleration.setLabel('bottom', 'Time (s)')
        ax_acceleration.plot(time, acceleration, pen='g')

        # Deception Probability plot
        ax_prob = export_widget.addPlot(title="Deception Probability Over Time", row=3, col=0)
        ax_prob.setLabel('left', 'Probability')
        ax_prob.setLabel('bottom', 'Time (s)')
        ax_prob.plot(time_data, deception_data, pen='m')

        # Export the graph as an image
        screenshot = export_widget.grab()
        screenshot.save(save_path, 'PNG')
        print(f"Graph image saved to: {save_path}")
    except Exception as e:
        print(f"Error exporting graph: {e}")

def export_csv(features, time_data, deception_data, output_file):
    """
    Export gaze data and corresponding deception probabilities to CSV.
    :param time_data: List of time values
    :param features: Normalized features (variance, velocity, acceleration)
    :param deception_data: Deception probabilities
    :param output_file: Output CSV file path
    """
    if len(time_data) == 0 or len(features) == 0:
        print("No data to export.")
        return

    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Time', 'Variance', 'Velocity', 'Acceleration', 'Deception Probability']) 
                         
        for i in range(len(time_data)):
            variance = features[i, 1] if i < len(features) else 0
            velocity = features[i, 2] if i < len(features) else 0
            acceleration = features[i, 3] if i < len(features) else 0
            deception = deception_data[i] if i < len(deception_data) else None

            writer.writerow([time_data[i], variance, velocity, acceleration, deception])

def export_features_csv(features, time_data, output_file):
    """
    Export gaze data and corresponding deception probabilities to CSV.
    :param time_data: List of time values
    :param features: Normalized features (variance, velocity, acceleration)
    :param deception_data: Deception probabilities
    :param output_file: Output CSV file path
    """
    if len(time_data) == 0 or len(features) == 0:
            print("No data to export.")
            return

    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Time', 'Variance', 'Velocity', 'Acceleration']) 
                        
        for i in range(len(time_data)):
            variance = features[i, 1] if i < len(features) else 0
            velocity = features[i, 2] if i < len(features) else 0
            acceleration = features[i, 3] if i < len(features) else 0

            writer.writerow([time_data[i], variance, velocity, acceleration])

def export_training_graph(features, transition_matrix, means, save_path):
    """
    Export training visualization graph (variance, velocity, acceleration, and transition matrix).
    :param features: Array of features [time, variance, velocity, acceleration].
    :param transition_matrix: HMM transition matrix.
    :param means: HMM state means.
    :param save_path: Path to save the graph image.
    """

    try:
        # Ensure export directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Create an exportable plot widget
        export_widget = pg.GraphicsLayoutWidget(show=False)
        export_widget.resize(800, 600)

        # Extract features
        time = features[:, 0]  # Time elapsed
        variance = features[:, 1]  # Normalized variance
        velocity = features[:, 2]  # Normalized velocity
        acceleration = features[:, 3]  # Normalized acceleration

        # Variance plot
        ax_variance = export_widget.addPlot(title="Variance Over Time")
        ax_variance.setLabel('left', 'Variance')
        ax_variance.setLabel('bottom', 'Time (s)')
        ax_variance.plot(time, variance, pen='r')

        # Velocity plot
        ax_velocity = export_widget.addPlot(title="Velocity Over Time", row=1, col=0)
        ax_velocity.setLabel('left', 'Normalized Velocity')
        ax_velocity.setLabel('bottom', 'Time (s)')
        ax_velocity.plot(time, velocity, pen='b')

        # Acceleration plot
        ax_acceleration = export_widget.addPlot(title="Acceleration Over Time", row=2, col=0)
        ax_acceleration.setLabel('left', 'Acceleration')
        ax_acceleration.setLabel('bottom', 'Time (s)')
        ax_acceleration.plot(time, acceleration, pen='g')

        # Export the graph as an image
        screenshot = export_widget.grab()
        screenshot.save(save_path, 'PNG')
        print(f"Graph image saved to: {save_path}")
    except Exception as e:
        print(f"Error exporting graph: {e}")