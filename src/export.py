import csv
import os
import math
import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import QApplication
from PyQt5 import QtCore

def _gaussian_pdf(x, mu, sigma):
    return (1.0 / (math.sqrt(2.0 * math.pi) * sigma)) * np.exp(-0.5 * ((x - mu) / sigma)**2)

def export_gmm_states_before_after(
    model_before,
    model_after,
    n_states=2,
    n_mix=1,
    n_features=2,   # or however many features you actually have
    save_path="exports/training.png"
):
    """
    Plots each state's GMM curves for 'before' and 'after' training in a 2×2 layout:
      - Top row: State 0  (Before vs After)
      - Bottom row: State 1 (Before vs After)
    Each subplot shows multiple curves (one per mixture-feature).
    """
    # Create an offscreen application if none exists
    app_created = False
    if not QApplication.instance():
        _ = QApplication([])
        app_created = True

    win = pg.GraphicsLayoutWidget(show=False)
    win.resize(1400, 800)

    # We'll keep domain from -3 to +3, sampling more points
    x_vals = np.linspace(-3, 3, 400)

    def make_sub_plot(state_idx, col_idx):
        title = f"State {state_idx} - {'Before' if col_idx == 0 else 'After'} Training"
        plt_item = win.addPlot(row=state_idx, col=col_idx, title=title)
        legend = plt_item.addLegend()
        legend.opts["labelTextSize"] = "8pt"
        legend.setOffset((10, 10))

        plt_item.setLabel('left', 'PDF')
        plt_item.setLabel('bottom', 'Feature Value')
        plt_item.setXRange(-3, 3)
        plt_item.showGrid(x=True, y=True, alpha=0.2)

        bottom_axis = plt_item.getAxis('bottom')
        bottom_axis.setTickSpacing(0.5, 0.1)
        return plt_item

    # For 2 states => 2×2 grid
    plots = [
        [make_sub_plot(0, 0), make_sub_plot(0, 1)],
        [make_sub_plot(1, 0), make_sub_plot(1, 1)],
    ]

    def get_params(model, st, mx, feat):
        mu = model.means_[st, mx, feat]
        var = model.covars_[st, mx, feat]
        return mu, np.sqrt(var)

    # Use different line styles for features
    feature_styles = [
        (QtCore.Qt.SolidLine, 'r'),   # For feature 0
        (QtCore.Qt.DashLine,  'g'),   # For feature 1
        # Add more if you have >2 features
    ]

    for st in range(n_states):
        for col_idx in (0, 1):
            plot_item = plots[st][col_idx]
            current_model = model_before if col_idx == 0 else model_after

            for mx in range(n_mix):
                for feat in range(min(n_features, len(feature_styles))):
                    mu, std = get_params(current_model, st, mx, feat)
                    y_vals = _gaussian_pdf(x_vals, mu, std)

                    style, color_str = feature_styles[feat]
                    label = f"Mix {mx}, Feat {feat}"
                    plot_item.plot(
                        x_vals, y_vals,
                        pen=pg.mkPen(color=color_str, width=2, style=style),
                        name=label
                    )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    screenshot = win.grab()
    screenshot.save(save_path, 'PNG')
    print(f"Exported GMM curves to: {save_path}")

    # if app_created:
    #     win.show()
    #     QApplication.instance().exec_()


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

def load_features_data(file_path):
    """
    We'll keep the same function name, but interpret columns as [Time, X, Y].
    Return them as x_arr, y_arr, t_arr in that order => (x_data,y_data,time_data).
    """
    x_list = []
    y_list = []
    t_list = []
    try:
        with open(file_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # row keys => "Time", "X", "Y"
                t_list.append(float(row["Time"]))
                x_list.append(float(row["X"]))
                y_list.append(float(row["Y"]))
    except Exception as e:
        print(f"Error reading features CSV: {e}")
        return [], [], []
    return np.array(x_list), np.array(y_list), np.array(t_list)

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

def export_training_graph(features, transition_matrix, means, save_path):
    """
    Similar approach for the training graph, focusing on 2 features + time. 
    """
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        export_widget = pg.GraphicsLayoutWidget(show=False)
        export_widget.resize(800, 600)

        t_f = features[:, 0]
        f1 = features[:, 1]
        f2 = features[:, 2]

        p1 = export_widget.addPlot(title="Feature1 Over Time")
        p1.setLabel('left', 'Feature1')
        p1.setLabel('bottom', 'Time (s)')
        p1.plot(t_f, f1, pen='r')

        p2 = export_widget.addPlot(title="Feature2 Over Time", row=1, col=0)
        p2.setLabel('left', 'Feature2')
        p2.setLabel('bottom', 'Time (s)')
        p2.plot(t_f, f2, pen='b')

        screenshot = export_widget.grab()
        screenshot.save(save_path, 'PNG')
        print(f"Graph image saved to: {save_path}")
    except Exception as e:
        print(f"Error exporting graph: {e}")