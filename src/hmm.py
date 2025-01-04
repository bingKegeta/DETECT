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
        # Assuming features are in the format [time, variance, velocity, fixation]
        time = features[:, 0]
        variance = features[:, 1]
        velocity = features[:, 2]
        fixation = features[:, 3]

        p1 = self.win.addPlot(title="Normalized Variance Over Time")
        p1.plot(time, variance, pen='r')
        p1.setLabel('left', "Normalized Variance")
        p1.setLabel('bottom', "Time (s)")

        p2 = self.win.addPlot(title="Normalized Velocity Over Time", row=1, col=0)
        p2.plot(time, velocity, pen='b')
        p2.setLabel('left', "Normalized Velocity")
        p2.setLabel('bottom', "Time (s)")

        p3 = self.win.addPlot(title="Fixation Over Time", row=2, col=0)
        p3.plot(time, fixation, pen='g')
        p3.setLabel('left', "Fixation")
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

    def train(self, x_data, y_data, time_data):
        """Train HMM using normalized and prepared features."""
        features = self.prepare_features(x_data, y_data, time_data)

        # Debug: Log feature statistics
        print(f"Feature Stats Before Training:")
        print(f"Mean: {np.mean(features, axis=0)}, Std: {np.std(features, axis=0)}")

        # Initialize transition matrix
        self.model.transmat_ = np.array([[0.9, 0.1], [0.3, 0.7]])
        print(f"Initial Transition Matrix: {self.model.transmat_}")

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

        print(f"Initialized Means: {self.model.means_}")
        
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
        - Fixation-based feature
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

        # Compute fixation-based feature
        # Quantize X and Y coordinates into a grid (e.g., 3x3 regions)
        grid_size = 3  # Adjust this as needed
        x_bins = np.linspace(np.min(x_data), np.max(x_data), grid_size + 1)
        y_bins = np.linspace(np.min(y_data), np.max(y_data), grid_size + 1)
        x_regions = np.digitize(x_data, x_bins) - 1  # Map X coordinates to regions
        y_regions = np.digitize(y_data, y_bins) - 1  # Map Y coordinates to regions
        fixation_regions = list(zip(x_regions, y_regions))

        # Compute the percentage of time spent in each region
        unique_regions, counts = np.unique(fixation_regions, axis=0, return_counts=True)
        total_time = len(fixation_regions)
        fixation_percentage = {tuple(region): count / total_time for region, count in zip(unique_regions, counts)}

        # Use the highest fixation percentage as the feature
        fixation = np.array([fixation_percentage.get(tuple(region), 0) for region in fixation_regions[:-1]])  # Shape: (N-1,)
        fixation_trimmed = fixation[:-1]  # Shape: (N-2,)

        # Normalize variance, velocity, and fixation feature using Standard Z-Score Normalization
        variance_trimmed = variance[:-1]  # Shape: (N-2,)
        velocity_trimmed = velocity[:-1]  # Shape: (N-2,)

        # Debug: Print raw features
        print(f"Raw Variance: {variance} | Shape: {variance.shape}")
        print(f"Raw Velocity: {velocity} | Shape: {velocity.shape}")
        print(f"Raw Fixation: {fixation} | Shape: {fixation.shape}")

        # Debug: Print trimmed features
        print(f"Trimmed variance: {variance_trimmed} | Shape: {variance_trimmed.shape}")
        print(f"Trimmed velocity: {velocity_trimmed} | Shape: {velocity_trimmed.shape}")
        print(f"Trimmed fixation: {fixation_trimmed} | Shape: {fixation_trimmed.shape}")

        # Standard Z-Score Normalization
        variance_mean = np.mean(variance_trimmed)
        variance_std = np.std(variance_trimmed)
        variance_norm = (variance_trimmed - variance_mean) / (variance_std if variance_std > 0 else 1)

        velocity_mean = np.mean(velocity_trimmed)
        velocity_std = np.std(velocity_trimmed)
        velocity_norm = (velocity_trimmed - velocity_mean) / (velocity_std if velocity_std > 0 else 1)

        fixation_mean = np.mean(fixation_trimmed)
        fixation_std = np.std(fixation_trimmed)
        fixation_norm = (fixation_trimmed - fixation_mean) / (fixation_std if fixation_std > 0 else 1)

        # Debug: Print normalization parameters and normalized features
        print(f"Variance Mean: {variance_mean}, Variance Std: {variance_std}")
        print(f"Velocity Mean: {velocity_mean}, Velocity Std: {velocity_std}")
        print(f"Fixation Mean: {fixation_mean}, Fixation Std: {fixation_std}")

        print(f"Normalized Variance: {variance_norm}")
        print(f"Normalized Velocity: {velocity_norm}")
        print(f"Normalized Fixation: {fixation_norm}")

        # Apply weights to normalized features
        variance_weighted = 0.5 * variance_norm  # Shape: (N-2,)
        velocity_weighted = 0.25 * velocity_norm  # Shape: (N-2,)
        fixation_weighted = 1.0 * fixation_norm  # Shape: (N-2,)

        # Debug: Print weighted features
        print(f"Weighted Variance: {variance_weighted}")
        print(f"Weighted Velocity: {velocity_weighted}")
        print(f"Weighted Fixation: {fixation_weighted}")

        # Align time data to match feature shapes
        aligned_time_data = time_data[2:]  # Shape: (N-2,)

        # **Ensure all feature arrays have the same length**
        assert aligned_time_data.shape[0] == variance_weighted.shape[0] == velocity_weighted.shape[0] == fixation_weighted.shape[0], \
            f"Feature lengths do not match: Time({aligned_time_data.shape[0]}), Variance({variance_weighted.shape[0]}), " \
            f"Velocity({velocity_weighted.shape[0]}), Fixation({fixation_weighted.shape[0]})"

        # Stack features together without additional slicing
        features = np.column_stack([
            aligned_time_data,      # Time elapsed
            variance_weighted,     # Normalized and weighted variance
            velocity_weighted,     # Normalized and weighted velocity
            fixation_weighted      # Normalized and weighted fixation feature
        ])  # Shape: (N-2, 4)

        # Debug: Print prepared features
        print(f"Prepared Weighted Features: {features}")

        return features
