import csv
from statistics import variance
import numpy as np

def load_baseline_data(file_path):
    """Load baseline CSV data and return x, y, and time arrays."""
    variance, velocity, acceleration, time_data = [], [], [], []
    with open(file_path, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            variance.append(float(row['Variance']))
            velocity.append(float(row['Velocity']))
            acceleration.append(float(row['Acceleration']))
            time_data.append(float(row['Time']))

        features = np.column_stack([
            time_data,
            variance,
            velocity,
            acceleration
        ])
    return features

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