import csv

def load_baseline_data(file_path):
    """Load baseline CSV data and return x, y, and time arrays."""
    x_data, y_data, time_data = [], [], []
    with open(file_path, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            x_data.append(float(row['X Coordinate']))
            y_data.append(float(row['Y Coordinate']))
            time_data.append(float(row['Time']))
    return x_data, y_data, time_data

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