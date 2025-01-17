from hmmlearn import hmm
import numpy as np
from PyQt5.QtWidgets import QVBoxLayout, QDialog
import pyqtgraph as pg

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
        # Assuming features are in the format [time, variance, velocity, acceleration]
        time = features[:, 0]
        variance = features[:, 1]
        velocity = features[:, 2]
        acceleration = features[:, 3]

        p1 = self.win.addPlot(title="Normalized Variance Over Time")
        p1.plot(time, variance, pen='r')
        p1.setLabel('left', "Normalized Variance")
        p1.setLabel('bottom', "Time (s)")

        p2 = self.win.addPlot(title="Normalized Velocity Over Time", row=1, col=0)
        p2.plot(time, velocity, pen='b')
        p2.setLabel('left', "Normalized Velocity")
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
            p4.setLabel('right', f"State {i} Mean: {mean}")

class HiddenMarkovModel:
    def __init__(self):
        self.model = hmm.GaussianHMM(n_components=2, covariance_type="diag", init_params='')  # Disable reinitialization

    def train(self, x_data, y_data, time_data):
        """Train HMM using normalized and prepared features."""
        features = self.prepare_features(x_data, y_data, time_data)

        # Initialize transition matrix
        self.model.transmat_ = np.array([[0.9, 0.1], [0.3, 0.7]])

        # Initialize means and covariances
        baseline_mean = np.mean(features, axis=0)
        baseline_std = np.std(features, axis=0)
        self.model.means_ = np.array([
            baseline_mean,  # State 0: Baseline behavior
            baseline_mean + 2 * baseline_std  # State 1: Deviations (abnormal behavior)
        ])
        self.model.covars_ = np.array([
            baseline_std**2,  # Variance for state 0
            (2 * baseline_std)**2  # Variance for state 1
        ])

        # Initialize start probabilities if not set
        if not hasattr(self.model, 'startprob_'):
            self.model.startprob_ = np.array([0.5, 0.5])

        # Train the HMM
        self.model.fit(features)
        print("HMM training completed successfully!")

        # Debug information
        print(f"Transition Matrix:\n{self.model.transmat_}")
        print(f"Means:\n{self.model.means_}")
        print(f"Covariances:\n{self.model.covars_}")

        # Show training visualization
        self.visualize_training(features, self.model.transmat_, self.model.means_)

    def visualize_training(self, features, transition_matrix, means):
        """Launch training visualization."""
        self.training_window = TrainingVisualization(features, transition_matrix, means)
        self.training_window.exec_()  # Show as a modal dialog

    def predict_proba(self, feature_vector):
        """Predict deception probability for new data."""
        feature = np.array(feature_vector).reshape(1, -1)  # Ensure it's (1,4)
        raw_probabilities = self.model.predict_proba(feature)

        # Apply bounds to avoid hard 0 or 1 probabilities
        bounded_probabilities = np.clip(raw_probabilities, 0.05, 0.95)
        print(f"Input Feature: {feature}")
        print(f"Raw State Probabilities: {raw_probabilities}")
        print(f"Bounded Probabilities: {bounded_probabilities}")
        return bounded_probabilities

    def prepare_features(self, x_data, y_data, time_data):
        """
        Prepare features for HMM training.
        Features include:
        - Time elapsed
        - Normalized variance of gaze positions
        - Normalized velocity of gaze movement
        - Acceleration of gaze movement
        """
        # Convert inputs to NumPy arrays if they aren't already
        x_data = np.asarray(x_data)
        y_data = np.asarray(y_data)
        time_data = np.asarray(time_data)

        if len(time_data) < 3:
            raise ValueError("Not enough data points for feature extraction.")

        # Compute variance over sliding windows of size 2
        # Ensure there are enough data points for sliding_window_view
        if len(x_data) < 2 or len(y_data) < 2:
            raise ValueError("Not enough data points for variance computation.")

        sliding_x = np.lib.stride_tricks.sliding_window_view(x_data, window_shape=2)
        sliding_y = np.lib.stride_tricks.sliding_window_view(y_data, window_shape=2)
        variance_x = np.var(sliding_x, axis=1)
        variance_y = np.var(sliding_y, axis=1)
        variance = variance_x + variance_y  # Shape: (N-1,)

        # Compute velocity
        delta_x = np.diff(x_data)  # Shape: (N-1,)
        delta_y = np.diff(y_data)  # Shape: (N-1,)
        delta_time = np.diff(time_data)  # Shape: (N-1,)
        
        # Prevent division by zero in velocity computation
        with np.errstate(divide='ignore', invalid='ignore'):
            velocity = np.sqrt(delta_x**2 + delta_y**2) / delta_time  # Shape: (N-1,)
            velocity = np.nan_to_num(velocity)  # Replace NaNs and infs with 0

        # Compute acceleration
        # np.diff(velocity) has shape (N-2,)
        # np.diff(time_data[1:]) also has shape (N-2,)
        delta_velocity = np.diff(velocity)  # Shape: (N-2,)
        delta_time_velocity = np.diff(time_data[1:])  # Shape: (N-2,)
        
        # Prevent division by zero in acceleration computation
        with np.errstate(divide='ignore', invalid='ignore'):
            acceleration = delta_velocity / delta_time_velocity  # Shape: (N-2,)
            acceleration = np.nan_to_num(acceleration)  # Replace NaNs and infs with 0

        # Normalize features
        # Adjust variance and velocity to match the shape of acceleration (N-2,)
        variance_trimmed = variance[:-1]  # Shape: (N-2,)
        velocity_trimmed = velocity[:-1]  # Shape: (N-2,)

        variance_max = np.max(variance_trimmed)
        variance_norm = variance_trimmed / variance_max if variance_max > 0 else variance_trimmed  # Shape: (N-2,)

        velocity_max = np.max(velocity_trimmed)
        velocity_norm = velocity_trimmed / velocity_max if velocity_max > 0 else velocity_trimmed  # Shape: (N-2,)

        acceleration_max = np.max(np.abs(acceleration))
        acceleration_norm = acceleration / acceleration_max if acceleration_max > 0 else acceleration  # Shape: (N-2,)

        # Align time data to match the shape of acceleration
        aligned_time_data = time_data[2:]  # Shape: (N-2,)

        # Stack the features together
        features = np.column_stack([aligned_time_data, variance_norm, velocity_norm, acceleration_norm])  # Shape: (N-2, 4)

        return features
