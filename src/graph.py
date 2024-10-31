# src/graph.py

from PyQt5 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg
import numpy as np
from scipy.signal import savgol_filter

WINDOW_LENGTH = 13
POLYORDER = 2

class GazeTrackingPlotter(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Real-Time Gaze Tracking")
        self.layout = QtWidgets.QVBoxLayout()
        self.setLayout(self.layout)

        # X and Y Line Plots
        self.plot_widget_x = pg.PlotWidget(title="X Coordinate Over Time")
        self.plot_widget_y = pg.PlotWidget(title="Y Coordinate Over Time")
        self.layout.addWidget(self.plot_widget_x)
        self.layout.addWidget(self.plot_widget_y)

        # 2D Scatter Plot
        self.plot_widget_2d = pg.PlotWidget(title="2D Gaze Points Over Time")
        self.layout.addWidget(self.plot_widget_2d)

        # 3D Scatter Plot (using pyqtgraph's GLViewWidget)
        self.plot_widget_3d = pg.opengl.GLViewWidget()
        self.scatter_3d = pg.opengl.GLScatterPlotItem()
        self.plot_widget_3d.addItem(self.scatter_3d)
        self.layout.addWidget(self.plot_widget_3d)

    def update_plots(self, time_data, x_data, y_data):
        # Apply smoothing
        x_data_smoothed = savgol_filter(x_data, WINDOW_LENGTH, POLYORDER) if len(x_data) >= WINDOW_LENGTH else x_data
        y_data_smoothed = savgol_filter(y_data, WINDOW_LENGTH, POLYORDER) if len(y_data) >= WINDOW_LENGTH else y_data

        # Update Line Plots
        self.plot_widget_x.clear()
        self.plot_widget_y.clear()
        self.plot_widget_x.plot(time_data, x_data_smoothed, pen='r')
        self.plot_widget_y.plot(time_data, y_data_smoothed, pen='b')

        # Update 2D Scatter Plot
        self.plot_widget_2d.clear()
        scatter_2d = pg.ScatterPlotItem(x=x_data, y=y_data, pen=None, brush=pg.intColor(200, 255))
        self.plot_widget_2d.addItem(scatter_2d)

        # Update 3D Scatter Plot
        colors = np.array([[1, 0, 0, 1]] * len(time_data))  # Red points
        pos = np.column_stack((x_data, y_data, time_data))
        self.scatter_3d.setData(pos=pos, color=colors, size=5)


def plot_gaze_tracking(time_data, x_data, y_data, ax_x=None, ax_y=None, scatter_ax=None, scatter3d_ax=None):
    app = QtWidgets.QApplication([])
    plotter = GazeTrackingPlotter()
    plotter.update_plots(time_data, x_data, y_data)
    plotter.show()
    app.exec_()


def plot_final_graphs(time_data, x_data, y_data, save_path='final_comprehensive_plots.png'):
    app = QtWidgets.QApplication([])

    # Create main widget to store final plots
    final_plotter = GazeTrackingPlotter()
    final_plotter.update_plots(time_data, x_data, y_data)

    # Capture and save the widget as an image
    screenshot = final_plotter.grab()
    screenshot.save(save_path, 'PNG')

    print(f"Final comprehensive plots saved as {save_path}.")
    final_plotter.show()
    app.exec_()
