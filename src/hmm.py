from PyQt5.QtWidgets import QVBoxLayout, QDialog
import pyqtgraph as pg
import numpy as np

class TrainingVisualization(QDialog):
    def __init__(self, features, transition_matrix, parent=None):
        super().__init__(parent)
        self.setWindowTitle("HMM Training Visualization")
        self.setMinimumSize(800, 600)
        
        layout = QVBoxLayout(self)
        self.win = pg.GraphicsLayoutWidget(show=True)
        layout.addWidget(self.win)

        # Plot features
        self.plot_features(features)

        # Plot transition matrix
        self.plot_transition_matrix(transition_matrix)

    def plot_features(self, features):
        # Assuming features are in the format [time, variance, velocity]
        time = features[:, 0]
        variance = features[:, 1]
        velocity = features[:, 2]

        p1 = self.win.addPlot(title="Variance Over Time")
        p1.plot(time, variance, pen='r')
        p1.setLabel('left', "Variance")
        p1.setLabel('bottom', "Time (s)")

        p2 = self.win.addPlot(title="Velocity Over Time", row=1, col=0)
        p2.plot(time, velocity, pen='b')
        p2.setLabel('left', "Velocity")
        p2.setLabel('bottom', "Time (s)")

    def plot_transition_matrix(self, transition_matrix):
        p3 = self.win.addPlot(title="Transition Matrix (Heatmap)", row=2, col=0)
        img = pg.ImageItem()
        img.setImage(transition_matrix)
        p3.addItem(img)
        p3.setLabel('left', "From State")
        p3.setLabel('bottom', "To State")



from hmmlearn import hmm
import numpy as np

class HiddenMarkovModel:
    def __init__(self):
        self.model = hmm.GaussianHMM(n_components=2, covariance_type="diag")  # Two states: high load, low load

    def train(self, x_data, y_data, time_data):
        """Train HMM using baseline data."""
        features = self.prepare_features(x_data, y_data, time_data)
        self.model.fit(features)
        print("HMM training completed successfully!")

        # Show training visualization
        transition_matrix = self.model.transmat_
        self.visualize_training(features, transition_matrix)

    def visualize_training(self, features, transition_matrix):
        """Launch training visualization."""
        self.training_window = TrainingVisualization(features, transition_matrix)
        self.training_window.exec_()  # Show as a modal dialog


    def predict_proba(self, x, y, timestamp):
        """Predict deception probability for new data."""
        feature = np.array([[timestamp, x, y]])
        probabilities = self.model.predict_proba(feature)
        print(f"Input Feature: {feature}")
        print(f"State Probabilities: {probabilities}")
        return probabilities

    def prepare_features(self, x_data, y_data, time_data):
        """
        Prepare features for HMM training.
        Features include:
        - Time elapsed
        - Variance of gaze positions
        - Velocity of gaze movement
        """
        if len(time_data) < 2:
            raise ValueError("Not enough data points for feature extraction.")

        # Compute variance over a sliding window
        variance_x = np.var(np.lib.stride_tricks.sliding_window_view(x_data, window_shape=2), axis=1)
        variance_y = np.var(np.lib.stride_tricks.sliding_window_view(y_data, window_shape=2), axis=1)
        variance = variance_x + variance_y

        # Compute velocity (distance moved per unit time)
        velocity = np.sqrt(np.diff(x_data)**2 + np.diff(y_data)**2) / np.diff(time_data)

        # Align time data to match the lengths of variance and velocity
        time_data_truncated = time_data[1:]  # Drop the first time value

        if len(variance) != len(velocity) or len(variance) != len(time_data_truncated):
            raise ValueError("Mismatched feature lengths.")

        # Stack the features together
        features = np.column_stack([time_data_truncated, variance, velocity])
        print(f"Features prepared: {features.shape[0]} rows, {features.shape[1]} columns")
        return features
