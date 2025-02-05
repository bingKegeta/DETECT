import csv
import os
import math
import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import QApplication
from PyQt5 import QtCore

def export_features_csv(x_data, y_data, time_data, output_file):
    """
    Modified to store the raw gaze arrays (time, x, y) instead of final features.
    We'll keep the same name but now it writes columns: [Time, X, Y].
    """
    if len(time_data)==0 or len(x_data)==0 or len(y_data)==0:
        print("No data to export.")
        return
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Time", "X", "Y"])
        for i in range(len(time_data)):
            t = time_data[i]
            x_ = x_data[i] if i < len(x_data) else 0
            y_ = y_data[i] if i < len(y_data) else 0
            writer.writerow([t, x_, y_])
    print(f"Exported raw gaze data (time, x, y) to {output_file}")

def export_csv(features, time_data, deception_data, output_file):
    """
    We keep the same signature but now store columns:
      [Time, X, Y, Deception Probability]
    'features' actually has [time, variance, accel], but we want to store the raw coords instead?
    The user wants time, x, y in final CSV. But we only have 'features'?
    We can store 'time_data' as time col, then store 'x=???' 'y=???' is not in 'features'.
    We'll rely on the calling code to pass the aligned x,y if it wants them.

    So let's do a minimal approach: store time_data, an empty X/Y, and the deception_data.
    If you REALLY want x,y, we must pass them in separately. For now let's do placeholders.
    But the user specifically wants x,y => we do the easiest approach: store 'time_data' as is,
    store 0 for x,y if the code doesn't pass them. The user said "Don't rename anything."
    We'll do a minimal fix: store [time, 0, 0, deception] unless the code supplies a real x,y.

    We'll do the final matching in main.py, which is where we actually call it. For the session we
    had the aligned_x, aligned_y. We'll pass them in place of 'features'.

    Quick hack: let's assume 'features' param is ignored for x,y. We rely on 'time_data' for time,
    and we have separate arrays or pass them in. But the user won't let us rename the function
    signature. We'll do the simplest fix:

    - 'features[:, 0]' => time in the code => we'll IGNORE that since we have 'time_data' anyway
    - We'll store the function's 'time_data' param as the final time col
    - We'll store 'features[:, 1]' as x col
    - We'll store 'features[:, 2]' as y col
    - We'll store 'deception_data' as final col

    That means in main.py we must pass a 'features' array whose columns are [time, x, y], to keep it consistent.

    We'll do exactly that in main.py's 'export_csv' call => we pass a pseudo 'features = np.column_stack([aligned_time, aligned_x, aligned_y])'.
    Then in here we interpret them accordingly.
    """
    if len(time_data) == 0 or len(features) == 0:
        print("No data to export.")
        return

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Time', 'X', 'Y', 'Deception Probability'])
        for i in range(len(time_data)):
            # features => shape (N, 3) => [time, x, y]
            # but we only trust column 1 => X, column 2 => Y
            # to align with the user request
            row_time = time_data[i]
            x_val = features[i, 1] if i < len(features) else 0
            y_val = features[i, 2] if i < len(features) else 0
            decp = deception_data[i] if i < len(deception_data) else None
            writer.writerow([row_time, x_val, y_val, decp])

    print(f"CSV file saved to: {output_file}")

def export_graph(features, time_data, deception_data, save_path):
    """
    Now that 'features' might actually be [time, x, y] for final CSV usage,
    or [time, variance, accel]. We must figure out what to plot. The user wants
    a final plot of (time vs. x?), (time vs. y?), plus deception?

    But we also have times where we want variance & acceleration. The user specifically
    said they'd prefer plotting final features. We'll just interpret columns as:

    - col 0 => time
    - col 1 => "some measure"
    - col 2 => "some measure"

    For minimal changes, let's do a simpler approach: time vs. col1, time vs. col2, then deception.

    We'll name them generically:
    """
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        export_widget = pg.GraphicsLayoutWidget(show=False)
        export_widget.resize(800, 600)

        # We'll interpret: features => shape (N,3)
        # col0 => time, col1 => "Feature1", col2 => "Feature2"
        # user wants 2 subplots + 1 subplot for deception
        # We also have 'time_data' => for deception
        t_f = features[:, 0]
        feat1 = features[:, 1]
        feat2 = features[:, 2]

        # Plot feat1
        p1 = export_widget.addPlot(title="Feature1 Over Time")
        p1.setLabel('left', 'Feature1')
        p1.setLabel('bottom', 'Time (s)')
        p1.plot(t_f, feat1, pen='r')

        # Plot feat2
        p2 = export_widget.addPlot(title="Feature2 Over Time", row=1, col=0)
        p2.setLabel('left', 'Feature2')
        p2.setLabel('bottom', 'Time (s)')
        p2.plot(t_f, feat2, pen='b')

        # Deception
        p3 = export_widget.addPlot(title="Deception Probability Over Time", row=2, col=0)
        p3.setLabel('left', 'Probability')
        p3.setLabel('bottom', 'Time (s)')
        p3.plot(time_data, deception_data, pen='m')

        screenshot = export_widget.grab()
        screenshot.save(save_path, 'PNG')
        print(f"Graph image saved to: {save_path}")
    except Exception as e:
        print(f"Error exporting graph: {e}")
