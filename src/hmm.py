import copy
import numpy as np
from hmmlearn.hmm import GMMHMM
from PyQt5.QtWidgets import QVBoxLayout, QDialog
import pyqtgraph as pg
import os

from src.export import export_training_graph, export_gmm_states_before_after

def log_summary_statistics(data, label):
    """Logs summary statistics for a dataset."""
    print(f"{label} Summary Statistics:")
    print(f"  Min: {np.min(data):.4f}, Max: {np.max(data):.4f}, Mean: {np.mean(data):.4f}, Median: {np.median(data):.4f}, Std: {np.std(data):.4f}")
          
def smooth(data, window_size=8):
    """Block averaging smoothing."""
    return np.convolve(data, np.ones(window_size) / window_size, mode='same')

class TrainingVisualization(QDialog):
    def __init__(self, features, transition_matrix, means, parent=None):
        super().__init__(parent)
        self.setWindowTitle("HMM Training Visualization")
        self.setMinimumSize(800, 600)

        layout = QVBoxLayout(self)
        self.win = pg.GraphicsLayoutWidget(show=True)
        layout.addWidget(self.win)

        self.plot_features(features)
        self.plot_transition_matrix(transition_matrix, means)

    def plot_features(self, features):
        # features = [time, variance, acceleration]
        time = features[:, 0]
        variance = features[:, 1]
        acceleration = features[:, 2]

        p1 = self.win.addPlot(title="Variance Over Time")
        p1.plot(time, variance, pen='r')
        p1.setLabel('left', "Variance")
        p1.setLabel('bottom', "Time (s)")

        p2 = self.win.addPlot(title="Acceleration Over Time", row=1, col=0)
        p2.plot(time, acceleration, pen='g')
        p2.setLabel('left', "Acceleration")
        p2.setLabel('bottom', "Time (s)")

    def plot_transition_matrix(self, transmat, means):
        p = self.win.addPlot(title="Transition Matrix (Heatmap)", row=2, col=0)
        img = pg.ImageItem()
        img.setImage(transmat)
        p.addItem(img)
        p.setLabel('left', "From State")
        p.setLabel('bottom', "To State")

        # Display means text
        for i, mean in enumerate(means):
            txt = pg.TextItem(f"State {i} Mean: {mean}", anchor=(0,1))
            txt.setPos(i, i)
            p.addItem(txt)


class HiddenMarkovModel:
    def __init__(self):
        """
        GMM-HMM:
          2 states, 2 mixtures each, diag covar
        """
        self.model = GMMHMM(
            n_components=2,
            n_mix=2,
            covariance_type="diag",
            n_iter=100,
            init_params='stmwc'   # Let model re-estimate startprob, transmat, means, weights, covars
        )

    def train(self, feature):
        # feature => shape (N, 3): [time, variance, acceleration]
        # remove time col => shape (N, 2)
        X = feature[:, 1:]

        print("Feature Stats Before Training:")
        print(f"Mean: {np.mean(X, axis=0)}, Std: {np.std(X, axis=0)}")

        # Provide an initial transmat
        self.model.transmat_ = np.array([
            [0.7, 0.3],
            [0.3, 0.7]
        ])
        print(f"Initial Transition Matrix: {self.model.transmat_}")

        baseline_mean = np.mean(X, axis=0)   # shape (2,)
        baseline_median = np.median(X, axis=0)  # shape (2,)
        baseline_std  = np.std(X, axis=0)    # shape (2,)

        # For 2 states x 2 mixtures x 2 features => shape is (2,2,2).
        # We'll initialize them fairly close:
        #   state0 mix0 = baseline
        #   state0 mix1 = baseline + small offset
        #   state1 mix0 = baseline + bigger offset
        #   state1 mix1 = baseline + bigger offset still
        means_init = np.zeros((2, 2, 2))
        # State 0:
        means_init[0, 0] = baseline_median
        means_init[0, 1] = baseline_median + 0.3 * baseline_std
        # State 1:
        means_init[1, 0] = baseline_mean + baseline_std
        means_init[1, 1] = baseline_mean + 0.5 * baseline_std

        self.model.means_ = means_init

        # Covars => shape (2,2,2) for diag
        # We'll keep them moderate
        covars_init = np.zeros((2, 2, 2))
        # state0 => smaller variance
        covars_init[0, 0] = (0.5 * baseline_std)**2
        covars_init[0, 1] = (0.8 * baseline_std)**2
        # state1 => bigger variance
        covars_init[1, 0] = (1.2 * baseline_std)**2
        covars_init[1, 1] = (1.5 * baseline_std)**2

        self.model.covars_ = covars_init

        # Keep a copy of the "before" model
        model_before = copy.deepcopy(self.model)

        # Fit
        self.model.fit(X)
        print("HMM training completed successfully!")
        print(f"Features shape: {X.shape}")
        print("Trained Transition Matrix:")
        print(self.model.transmat_)
        print("Trained Means:")
        print(self.model.means_)
        print("Trained Covariances:")
        print(self.model.covars_)

        # Export before/after
        export_gmm_states_before_after(
            model_before, self.model,
            n_states=2, n_mix=2, n_features=2,
            save_path="exports/training.png"
        )

        self.visualize_training(feature, self.model.transmat_, self.model.means_)

    def predict_proba(self, feature):
        # again, remove time => shape (N,2)
        X = feature[:, 1:]
        raw_prob = self.model.predict_proba(X)
        log_summary_statistics(raw_prob, "Raw Probabilities")
        return raw_prob

    def prepare_features(self, x_data, y_data, time_data):
        """
        Convert raw x,y,t => [time, variance, acceleration].
        """
        x_data = np.asarray(x_data)
        y_data = np.asarray(y_data)
        time_data = np.asarray(time_data)

        if len(time_data) < 3:
            raise ValueError("Not enough data points for feature extraction.")

        # variance
        slide_x = np.lib.stride_tricks.sliding_window_view(x_data, 2)
        slide_y = np.lib.stride_tricks.sliding_window_view(y_data, 2)
        var_x = np.var(slide_x, axis=1)
        var_y = np.var(slide_y, axis=1)
        variance = var_x + var_y  # shape => (N-1,)

        # velocity => needed for acceleration
        dx = np.diff(x_data)
        dy = np.diff(y_data)
        dt = np.diff(time_data)
        with np.errstate(divide='ignore', invalid='ignore'):
            velocity = np.sqrt(dx**2 + dy**2) / dt
        velocity = np.nan_to_num(velocity)

        # acceleration
        with np.errstate(divide='ignore', invalid='ignore'):
            acceleration = np.diff(velocity) / np.diff(time_data[:-1])
        acceleration = np.abs(np.nan_to_num(acceleration))

        # align => shape => (N-2,)
        variance_trim = variance[:-1]

        # # z-score
        # var_mean, var_std = np.mean(variance_trim), np.std(variance_trim)
        # variance_norm = (variance_trim - var_mean) / (var_std if var_std>0 else 1)

        # acc_mean, acc_std = np.mean(acceleration), np.std(acceleration)
        # acceleration_norm = (acceleration - acc_mean) / (acc_std if acc_std>0 else 1)

        # min-max
        var_min, var_max = np.min(variance_trim), np.max(variance_trim)
        variance_norm = (variance_trim - var_min) / (var_max - var_min)

        acc_min, acc_max = np.min(acceleration), np.max(acceleration)
        acceleration_norm = (acceleration - acc_min) / (acc_max - acc_min)

        # # smoothing
        # variance_smooth = smooth(variance_norm, 8)
        # acceleration_smooth = smooth(acceleration_norm, 12)

        variance_smooth = variance_norm
        acceleration_smooth = acceleration_norm

        # aligned time => (N-2,)
        aligned_time = time_data[2:]

        # final => shape (N-2,3)
        features = np.column_stack([
            aligned_time,
            variance_smooth,
            acceleration_smooth
        ])

        # debug
        log_summary_statistics(variance, "Raw Variance")
        log_summary_statistics(velocity, "Velocity (internal only)")
        log_summary_statistics(acceleration, "Raw Acceleration")
        log_summary_statistics(variance_norm, "Norm Variance")
        log_summary_statistics(acceleration_norm, "Norm Acceleration")
        log_summary_statistics(variance_smooth, "Smooth Variance")
        log_summary_statistics(acceleration_smooth, "Smooth Acceleration")

        print(f"Prepared Features:\n{features}")
        return features

    def visualize_training(self, features, transmat, means):
        export_training_graph(features, transmat, means, os.path.join("exports", "baseline.png"))
        tv = TrainingVisualization(features, transmat, means)
        tv.exec_()
