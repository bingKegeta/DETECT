# HMM Branch - Hidden Markov Model for Deception Detection

## Overview
This branch introduces a Hidden Markov Model (HMM) for detecting deception based on gaze tracking data. It incorporates the following features:
- **HMM Model**: Trained using baseline gaze data (variance and acceleration) to classify states (e.g., truthful vs. deceptive).
- **PyQt Training Visualization**: Visual representation of model training, including feature distributions and transition probabilities.
- **Deception Probability Computation**: Calculates probabilities for session data and includes them in the exported CSV.
- **Gaussian Mixture Emission**: Exported visualizations for Gaussian emission probabilities.

## Key Features

### Flexible Baseline Handling
- **Baseline Video**: Process a baseline video to generate x, y, and time data for training.
- **Baseline CSV**: Directly use a preprocessed baseline CSV file containing x, y, and time data.
- Configuration handled via `config.json`.

### Session Data Analysis
- Processes session videos to compute gaze data and deception probabilities.
- Exports results in CSV format with `Time`, `X Coordinate`, `Y Coordinate`, and `Deception Probability`.

### Gaussian Mixture Model (GMM-HMM)
- **Number of Mixtures**: Configurable in `src/hmm.py`.
- **Transition Matrix**: Adjustable to reflect state transition probabilities.
- **Emission Parameters**: Gaussian curves adapt to training data and are exported as `training.png`.

### Feature Processing
- **Features**: Variance and acceleration (velocity removed).
- **Normalization**: Min-max normalization (0 to 1) applied to both variance and absolute acceleration.
- **Smoothing**: Stronger smoothing applied to acceleration to reduce noise.

### Exports
1. **`baseline.csv`**: Stores x, y, and time data from the baseline video.
2. **`baseline.png`**: Visualization of baseline feature distributions and HMM training results.
3. **`session.csv`**: Stores x, y, and time data from the session video.
4. **`gaze_data.csv`**: Processed session data, including deception probabilities.
5. **`final_comprehensive_plots.png`**: Graph summarizing variance, acceleration, and deception probabilities over time.
6. **`training.png`**: Gaussian curves for state emission probabilities before and after training.

## Example `config.json`
```json
{
  "source": "video",
  "baseline_video": "test_media/baseline_video.mp4",
  "baseline_csv": "exports/baseline.csv",
  "session_video": "test_media/session_video.mp4",
  "session_csv": "exports/session.csv",
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

### Explanation of Parameters

#### Main Parameters
- **`source`**: Input source for gaze tracking. Options:
  - `"webcam"`: Real-time tracking using the webcam.
  - `"video"`: Analysis of a video file.
  - `"image"`: Analysis of a single image.

#### Session Parameters
- **`session_video`**: Path to the session video file.
- **`session_csv`**: Path to an existing session CSV file. If provided, x, y, and time data will be reprocessed into features.

#### Baseline Parameters
- **`baseline`**: (`true`/`false`) Indicates whether baseline data is used for training.
- **`baseline_video`**: Path to the baseline video file.
- **`baseline_csv`**: Path to an existing baseline CSV file.

#### Export Options
- **`csv`**: (`true`/`false`) Exports gaze data to a CSV file.
- **`graph`**: (`true`/`false`) Exports graphs to an image file.
- **`animation`**: (`true`/`false`) Exports animations (if implemented).
- **`export_dir`**: Directory for saving exported files.

#### Processing Options
- **`dot_display`**: Displays detected gaze points for visualization.
- **`categorize`**: Categorizes gaze direction (e.g., left, center, right).
- **`graph`**: Displays real-time graphs of gaze data.
- **`affine`**: Applies affine transformations to stabilize gaze coordinates.

### Updating HMM Parameters
- **Number of Mixtures**: Change `n_mix` in `src/hmm.py`.
- **Transition Matrix**: Update `self.model.transmat_` in `src/hmm.py`.
- **Emission Means and Covariances**: Adjust initialization logic in `src/hmm.py`.

## Usage
1. Modify `config.json` with the desired settings.
2. Run the application:
   ```bash
   python main.py config.json
   ```
3. View exported CSV and images in the specified output directory.

