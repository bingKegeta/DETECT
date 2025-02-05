# Real-Time Deterministic Deception Analysis

This repository implements a **deterministic algorithm** for computing deception probabilities in real time using gaze-tracking data. Below is an overview of the key files, the approach taken, and how to configure the application.

---

## Core Files

1. **`analysis.py`**  
   - Defines a simple `Analysis` class that processes **one gaze point at a time** (`time, x, y`) and returns:
     1. A scaled “variance” measure (distance squared from the last point),
     2. A scaled “acceleration” measure, and
     3. A **final deception probability** in `[0.01..0.95]`.
   - Internally keeps minimal state (`last_x, last_y, last_time, last_velocity`) so each frame can be processed in isolation.
   - Uses **clipping** (e.g., `[0..10]`) and a linear transform to map each measure into `[0.01..0.95`.
   
2. **`main.py`**  
   - Demonstrates real-time capturing of gaze data (either from **webcam** or **session video** or a **CSV** of `(x, y, time)` points).
   - On each new gaze point, calls `Analysis.single_update(time, x, y)` to compute scaled features plus a probability.
   - Accumulates those features (`variance_norm`, `acceleration_norm`) along with the final probability in arrays, so you can export them at session end.

3. **Other Utility Files**  
   - `export.py`, `utils.py`, etc. handle exporting CSVs, graphs, and any needed configuration loading.  
   - `args.py` or `config.json` parse the user’s input or define additional runtime parameters.

---

## Configuration (`config.json`)

A typical `config.json` might look like:

```json
{
  "source": "video",
  "session_video": "my_session.mp4",
  "session_csv": null,
  "dot_display": false,
  "affine": false,
  "categorize": false,
  "export_dir": "./exports",
  "export": {
    "csv": true,
    "graph": true,
    "animation": false
  }
}
