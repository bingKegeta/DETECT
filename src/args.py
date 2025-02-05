import argparse
import json
import os

def parse_args():
    parser = argparse.ArgumentParser(
        description="Gaze tracking with real-time or offline input and CSV export."
    )
    parser.add_argument(
        '--source', 
        type=str, 
        required=True, 
        choices=['webcam', 'image', 'video'],
        help="Data source for gaze detection: 'webcam', 'image', or 'video'."
    )
    parser.add_argument(
        '--path', 
        type=str, 
        help="Path to image or video if source != 'webcam'."
    )
    parser.add_argument(
        '--graph', 
        action='store_true', 
        help="Show a real-time graph of gaze coordinates."
    )
    parser.add_argument(
        '--affine', 
        action='store_true', 
        help="Apply affine transformation for face stabilization."
    )
    parser.add_argument(
        '--csv', 
        type=str, 
        help="Export gaze data to a CSV file."
    )
    parser.add_argument(
        '--csv_interval', 
        type=float, 
        default=1.0,
        help="Time interval in seconds for exporting CSV data."
    )
    return parser.parse_args()

def load_config(json_file_path):
    """
    Simplified config loader: no baseline references. We only handle:
      - source (webcam, image, video)
      - session_video, session_csv
      - export => { csv, graph, animation }
      - export_dir
      - dot_display, categorize, affine, etc.
    """
    if not os.path.exists(json_file_path):
        raise FileNotFoundError(f"Configuration file {json_file_path} does not exist.")
    
    with open(json_file_path, 'r') as file:
        config = json.load(file)

    # required 'source' check
    if 'source' not in config:
        raise ValueError("Config missing required 'source' field.")
    if config['source'] not in ['webcam', 'image', 'video']:
        raise ValueError("Invalid 'source'. Must be 'webcam', 'image', or 'video'.")

    # optional fields, default to None
    config.setdefault('session_video', None)
    config.setdefault('session_csv', None)

    # if user sets 'source' != 'webcam', we expect at least session_video or session_csv
    if config['source'] != 'webcam':
        if not (config['session_video'] or config['session_csv']):
            raise ValueError("Must provide 'session_video' or 'session_csv' when source != 'webcam'.")

    # Check mandatory 'dot_display' param
    if 'dot_display' not in config:
        raise ValueError("Config missing 'dot_display' boolean.")

    # handle 'export' dictionary
    if 'export' in config:
        export_opts = config['export']
        if not isinstance(export_opts, dict):
            raise ValueError("'export' must be a dictionary with 'csv' and 'graph' keys.")
        if 'csv' not in export_opts or 'graph' not in export_opts:
            raise ValueError("'export' dictionary must have 'csv' and 'graph' keys.")
        if (export_opts['csv'] or export_opts['graph'] or export_opts.get('animation',False)):
            if 'export_dir' not in config:
                raise ValueError("'export_dir' required if export.csv or export.graph is true.")

            if not os.path.isdir(config['export_dir']):
                os.makedirs(config['export_dir'], exist_ok=True)

    else:
        # no export info => default
        config['export'] = {'csv': False, 'graph': False, 'animation': False}

    # Additional booleans
    config.setdefault('affine', False)
    config.setdefault('categorize', False)

    return config
