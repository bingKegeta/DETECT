# HMM Branch - Hidden Markov Model for Deception Detection

## Overview
This branch introduces a Hidden Markov Model (HMM) for detecting deception based on gaze tracking data. It incorporates the following features:
- **HMM Model**: Trained using baseline gaze data (variance, velocity, and time) to classify states (e.g., truthful vs. deceptive).
- **PyQt Training Visualization**: Visual representation of model training, including variance, velocity, and transition probabilities.
- **Deception Probability Computation**: Calculates probabilities for session data and includes them in the exported CSV.

## Key Features
1. **Flexible Baseline Handling**:
   - Can process a baseline video and generate CSV data for training.
   - Alternatively, directly use a preprocessed baseline CSV.
   - Configuration handled via `config.json`.

2. **Session Data Analysis**:
   - Processes session videos to compute gaze data and deception probabilities.
   - Exports results in CSV format with `Time`, `X Coordinate`, `Y Coordinate`, and `Deception Probability`.

3. **Graphical Visualization**:
   - Real-time graphs for gaze data (`X`, `Y` coordinates).
   - Training visualization includes variance, velocity, and HMM transition matrix.

## Example `config.json`
```json
{
  "source": "video",
  "path": "test_media/session_video.mp4",
  "baseline_video": "test_media/baseline_video.mp4",
  "baseline_csv": "exports/baseline.csv",
  "export": {
    "csv": true,
    "graph": true,
    "animation": true
  },
  "export_dir": "./exports",
  "dot_display": false,
  "categorize": true,
  "graph": true,
  "affine": true,
  "csv_interval": 1.0,
  "baseline": true
}
```


## `config.json` Parameters

The `config.json` file contains configuration options for running the application. Below is a breakdown of the parameters:

### Main Parameters
- **`source`**: Specifies the input source for gaze tracking. Options:
  - `"webcam"`: Uses the webcam for real-time tracking.
  - `"video"`: Uses a video file for analysis.
  - `"image"`: Uses an image file for analysis.

- **`path`**: Path to the session video or image file to be processed (e.g., `"test_media/session_video.mp4"`).

### Baseline Parameters
- **`baseline`**: (`true`/`false`) Indicates whether baseline data is used for training the HMM model.
- **`baseline_video`**: Path to the baseline video file (optional). If provided, the video is processed to generate a baseline CSV file.
- **`baseline_csv`**: Path to an existing baseline CSV file. Used if `baseline_video` is not provided.

### Export Options
- **`export`**: Object containing export preferences:
  - **`csv`**: (`true`/`false`) Exports gaze data and deception probabilities to a CSV file.
  - **`graph`**: (`true`/`false`) Exports gaze data graphs to an image file.
  - **`animation`**: (`true`/`false`) Exports a real-time animation of gaze tracking (if implemented).

- **`export_dir`**: Directory where export files (CSV, graphs) will be saved.

### Processing Options
- **`dot_display`**: (`true`/`false`) Displays dots over detected gaze points for visualization.
- **`categorize`**: (`true`/`false`) Categorizes gaze direction (e.g., left, center, right) for debugging or analysis.
- **`graph`**: (`true`/`false`) Displays real-time graphs of gaze data during processing.
- **`affine`**: (`true`/`false`) Applies affine transformations to stabilize gaze coordinates.

### Session Parameters
- **`csv_interval`**: Interval (in seconds) for saving gaze data to the CSV file.
- **`baseline`**: (`true`/`false`) Enables baseline processing and model training using either `baseline_video` or `baseline_csv`.

## Updated Files
1. **`main.py`**: Manages the overall flow, baseline setup, and session processing.
2. **`process.py`**: Processes video frames to extract gaze data.
3. **`hmm.py`**: Implements the HMM model and training visualization.
4. **`export.py`**: Updates CSV export to include deception probabilities.
5. **`utils.py`**: Utility functions for data handling.

## Usage
1. Modify `config.json` with the desired settings.
2. Run the application:
   ```bash
   python main.py config.json
   ```
3. View training visualization and exported CSV in the specified output directory.