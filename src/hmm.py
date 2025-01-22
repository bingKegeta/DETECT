from cv2 import log
from hmmlearn import hmm
import numpy as np
from PyQt5.QtWidgets import QVBoxLayout, QDialog
import pyqtgraph as pg
from src.export import export_training_graph
import os


def log_summary_statistics(data, label):
    """Logs summary statistics for a dataset."""
    print(f"{label} Summary Statistics:")
    print(f"  Min: {np.min(data):.4f}, Max: {np.max(data):.4f}, Mean: {np.mean(data):.4f}, Std: {np.std(data):.4f}")

def smooth(data, window_size=5):
    """
    Smooth data using block averaging.
    Args:
        data (np.ndarray): Input data to smooth.
        window_size (int): Size of the smoothing block.
    Returns:
        np.ndarray: Smoothed data.
    """
    smoothed_data = np.convolve(data, np.ones(window_size) / window_size, mode='same')

    return smoothed_data

class TrainingVisualization(QDialog):
    def __init__(self, features, transition_matrix, means, parent=None):
        super().__init__(parent)
        self.setWindowTitle("HMM Training Visualization")
        self.setMinimumSize(800, 600)

        layout = QVBoxLayout(self)
        self.win = pg.GraphicsLayoutWidget(show=True)
        layout.addWidget(self.win)

        # Plot features
        self.plot_features(features)

        # Plot transition matrix
        self.plot_transition_matrix(transition_matrix, means)

    def plot_features(self, features):
        # Assuming features are in the format [time, variance, velocity, fixation]
        time = features[:, 0]
        variance = features[:, 1]
        velocity = features[:, 2]
        acceleration = features[:, 3]

        p1 = self.win.addPlot(title="Variance Over Time")
        p1.plot(time, variance, pen='r')
        p1.setLabel('left', "Variance")
        p1.setLabel('bottom', "Time (s)")

        p2 = self.win.addPlot(title="Velocity Over Time", row=1, col=0)
        p2.plot(time, velocity, pen='b')
        p2.setLabel('left', "Velocity")
        p2.setLabel('bottom', "Time (s)")

        p3 = self.win.addPlot(title="Acceleration Over Time", row=2, col=0)
        p3.plot(time, acceleration, pen='g')
        p3.setLabel('left', "Acceleration")
        p3.setLabel('bottom', "Time (s)")

    def plot_transition_matrix(self, transition_matrix, means):
        p4 = self.win.addPlot(title="Transition Matrix (Heatmap)", row=3, col=0)
        img = pg.ImageItem()
        img.setImage(transition_matrix)
        p4.addItem(img)
        p4.setLabel('left', "From State")
        p4.setLabel('bottom', "To State")

        # Display means as annotations
        for i, mean in enumerate(means):
            text = pg.TextItem(f"State {i} Mean: {mean}", anchor=(0, 1))
            text.setPos(i, i)
            p4.addItem(text)

class HiddenMarkovModel:
    def __init__(self):
        self.model = hmm.GaussianHMM(n_components=2, covariance_type="diag", init_params='')  # Disable reinitialization

    def train(self, features):
        """Train HMM using normalized and prepared features."""

        feature = np.array(features).reshape(1, -1)  # Ensure it's (1,4)

        # Remove the time column (assumed to be the first column)
        features = features[:, 1:]  # Retain only variance, velocity, and acceleration

        # Debug: Log feature statistics
        print(f"Feature Stats Before Training:")
        print(f"Mean: {np.mean(features, axis=0)}, Std: {np.std(features, axis=0)}")

        # Initialize transition matrix
        self.model.transmat_ = np.array([[0.6, 0.4], [0.4, 0.6]])
        print(f"Initial Transition Matrix: {self.model.transmat_}")

        # Initialize means and covariances
        baseline_mean = np.mean(features, axis=0)
        baseline_std = np.std(features, axis=0)
        self.model.means_ = np.array([
            baseline_mean + 2 * baseline_std,  # State 0: Baseline behavior
            baseline_mean  # State 1: Deviations (abnormal behavior)
        ])
        self.model.covars_ = np.array([
            (2 * baseline_std)**2,  # Variance for state 0
            baseline_std**2  # Variance for state 1
        ])

        # Initialize start probabilities if not set
        if not hasattr(self.model, 'startprob_'):
            self.model.startprob_ = np.array([0.5, 0.5])

        print(f"Initialized Means: {self.model.means_}")
        
        # Train the HMM
        self.model.fit(features)
        print("HMM training completed successfully!")

        # Debug information
        print(f"Features: {features.shape}")
        print(f"Transition Matrix:\n{self.model.transmat_}")
        print(f"Means:\n{self.model.means_}")
        print(f"Covariances:\n{self.model.covars_}")

        # Show training visualization
        self.visualize_training(feature, self.model.transmat_, self.model.means_)

    def visualize_training(self, features, transition_matrix, means):
        """Launch training visualization."""
        export_training_graph(features, transition_matrix, means, os.path.join("./exports", "baseline.png"))
        self.training_window = TrainingVisualization(features, transition_matrix, means)
        self.training_window.exec_()  # Show as a modal dialog

    def predict_proba(self, feature_vector):
        """Predict deception probability for new data."""
        feature = np.array(feature_vector).reshape(1, -1)  # Ensure it's (1,4)
        features = feature[:, 1:]  # Retain only variance, velocity, and acceleration
        raw_probabilities = self.model.predict_proba(features)

        # Apply bounds to avoid hard 0 or 1 probabilities
        bounded_probabilities = np.clip(raw_probabilities, 0.05, 0.95)
        # print(f"Input Feature: {feature}")
        # print(f"Raw State Probabilities: {raw_probabilities}")
        # print(f"Bounded Probabilities: {bounded_probabilities}")
        
        return raw_probabilities

    def prepare_features(self, x_data, y_data, time_data):
        """
        Prepare features for HMM training.
        Features include:
        - Time elapsed
        - Normalized variance of gaze positions
        - Normalized velocity of gaze movement
        - Normalized acceleration (derivative of velocity)
        """
        # Convert inputs to NumPy arrays if they aren't already
        x_data = np.asarray(x_data)
        y_data = np.asarray(y_data)
        time_data = np.asarray(time_data)

        if len(time_data) < 3:
            raise ValueError("Not enough data points for feature extraction.")

        # Compute variance over sliding windows of size 2
        sliding_x = np.lib.stride_tricks.sliding_window_view(x_data, window_shape=2)
        sliding_y = np.lib.stride_tricks.sliding_window_view(y_data, window_shape=2)
        variance_x = np.var(sliding_x, axis=1)
        variance_y = np.var(sliding_y, axis=1)
        variance = variance_x + variance_y  # Shape: (N-1,)

        # Compute velocity
        delta_x = np.diff(x_data)  # Shape: (N-1,)
        delta_y = np.diff(y_data)  # Shape: (N-1,)
        delta_time = np.diff(time_data)  # Shape: (N-1,)

        with np.errstate(divide='ignore', invalid='ignore'):
            velocity = np.sqrt(delta_x**2 + delta_y**2) / delta_time  # Shape: (N-1,)
            velocity = np.nan_to_num(velocity)  # Replace NaNs and infs with 0

        # Compute acceleration (derivative of velocity)
        acceleration = np.diff(velocity) / np.diff(time_data[:-1])  # Shape: (N-2,)
        acceleration = np.nan_to_num(acceleration)  # Replace NaNs and infs with 0

        # Normalize variance, velocity, and acceleration using Standard Z-Score Normalization
        variance_trimmed = variance[:-1]  # Shape: (N-2,)
        velocity_trimmed = velocity[:-1]  # Shape: (N-2,)

        # Standard Z-Score Normalization
        variance_mean = np.mean(variance_trimmed)
        variance_std = np.std(variance_trimmed)
        variance_norm = (variance_trimmed - variance_mean) / (variance_std if variance_std > 0 else 1)

        velocity_mean = np.mean(velocity_trimmed)
        velocity_std = np.std(velocity_trimmed)
        velocity_norm = (velocity_trimmed - velocity_mean) / (velocity_std if velocity_std > 0 else 1)

        acceleration_mean = np.mean(acceleration)
        acceleration_std = np.std(acceleration)
        acceleration_norm = (acceleration - acceleration_mean) / (acceleration_std if acceleration_std > 0 else 1)

        variance_smooth = smooth(variance_norm)
        velocity_smooth = smooth(velocity_norm)
        acceleration_smooth = smooth(acceleration_norm)

        # Debug: Log feature statistics
        log_summary_statistics(variance, "Raw Variance")
        log_summary_statistics(velocity, "Raw Velocity")
        log_summary_statistics(acceleration, "Raw Acceleration")

        log_summary_statistics(variance_norm, "Normalized Variance")
        log_summary_statistics(velocity_norm, "Normalized Velocity")
        log_summary_statistics(acceleration_norm, "Normalized Acceleration")

        log_summary_statistics(variance_smooth, "Smoothed Variance")
        log_summary_statistics(velocity_smooth, "Smoothed Velocity")
        log_summary_statistics(acceleration_smooth, "Smoothed Acceleration")

        # Align time data to match feature shapes
        aligned_time_data = time_data[2:]  # Shape: (N-2,)

        # **Ensure all feature arrays have the same length**
        assert aligned_time_data.shape[0] == variance_smooth.shape[0] == velocity_smooth.shape[0] == acceleration_smooth.shape[0], \
            f"Feature lengths do not match: Time({aligned_time_data.shape[0]}), Variance({variance_smooth.shape[0]}), " \
            f"Velocity({velocity_smooth.shape[0]}), Acceleration({acceleration_smooth.shape[0]})"

        # Stack features together without additional slicing
        features = np.column_stack([
            aligned_time_data,      # Time elapsed
            variance_smooth,        # Smoothed variance
            velocity_smooth,        # Smoothed velocity
            acceleration_smooth     # Smoothed acceleration
        ])  # Shape: (N-2, 4)

        # Debug: Print prepared features
        print(f"Prepared Smoothed Features: {features}")

        return features