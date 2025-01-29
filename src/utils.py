import csv
import numpy as np

def load_features_data(file_path):
    """
    Load CSV data with columns [Time, X, Y] and return x_data, y_data, time_data arrays.
    """
    x_data, y_data, t_data = [], [], []
    with open(file_path, 'r', newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            # Expect columns "Time", "X", "Y"
            t_data.append(float(row["Time"]))
            x_data.append(float(row["X"]))
            y_data.append(float(row["Y"]))

    return np.array(x_data), np.array(y_data), np.array(t_data)

def HorizontalRegion(x: float) -> str:
    if x < 0.033:
        return "Right"
    elif x > 0.037:
        return "Left"
    else:
        return "Center"

def VerticalRegion(y: float) -> str:
    if y > 0.073:
        return "Up"
    elif y < 0.069:
        return "Down"
    else:
        return "Center"
