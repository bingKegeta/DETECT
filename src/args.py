import argparse
import json
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Gaze and pupil tracking with smoothing, graphing, and CSV export.")
    
    parser.add_argument('--source', type=str, required=True, choices=['webcam', 'image', 'video'],
                        help="Source type for gaze detection: 'webcam', 'image', or 'video'.")
    
    parser.add_argument('--path', type=str, help="Path to image or video file, required if source is not 'webcam'.")
    
    parser.add_argument('--graph', action='store_true', help="Show a real-time graph of gaze coordinates.")
    
    parser.add_argument('--affine', action='store_true', help="Apply affine transformation for face stabilization.")
    
    parser.add_argument('--csv', type=str, help="Export gaze data to a CSV file. Provide file path.")\
    
    parser.add_argument('--csv_interval', type=float, default=1.0,
                        help="Time interval in seconds for exporting CSV data.")
    
    return parser.parse_args()

def load_config(json_file_path):
    """
    Load configuration settings from a JSON file, including handling the 'export' option with sub-requirements
    for 'csv' and 'graph' outputs.

    Args:
        json_file_path (str): The path to the configuration JSON file.

    Returns:
        dict: A dictionary containing the configuration settings.
    """
    if not os.path.exists(json_file_path):
        raise FileNotFoundError(f"The configuration file {json_file_path} does not exist.")
    
    with open(json_file_path, 'r') as file:
        config = json.load(file)
    
    # Validate required fields
    required_fields = ['source']
    for field in required_fields:
        if field not in config:
            raise ValueError(f"The configuration is missing the required '{field}' field.")
    
    # Check choices for 'source'
    if config['source'] not in ['webcam', 'image', 'video']:
        raise ValueError("Invalid value for 'source'. Must be 'webcam', 'image', or 'video'.")
    
    # Handle optional fields with defaults
    config.setdefault('path', None)
    config.setdefault('graph', True)
    config.setdefault('affine', False)
    config.setdefault('csv_interval', 1.0)
    config.setdefault('categorize', False)
    
    # If source is not 'webcam', 'path' is required
    if config['source'] != 'webcam' and not config['path']:
        raise ValueError("The 'path' field is required when 'source' is not 'webcam'.")
    
    # If dot display is not given
    if 'dot_display' not in config:
        raise ValueError("The \"dot_display\" parameter (boolean) is not given")
    
    # Handle 'export' sub-requirements
    if 'export' in config:
        export_options = config['export']
        if not isinstance(export_options, dict):
            raise ValueError("The 'export' field must be a dictionary with 'csv' and 'graph' options.")
        
        # Validate 'export' dictionary structure
        if 'csv' not in export_options or 'graph' not in export_options:
            raise ValueError("The 'export' dictionary must contain both 'csv' and 'graph' keys.")
        
        # If any export is true, ensure export_dir is provided
        if export_options['csv'] or export_options['graph'] or export_options['animation']:
            if 'export_dir' not in config:
                raise ValueError("The 'export_dir' field is required when 'export.csv' or 'export.graph' is enabled.")
            
            export_dir = config['export_dir']
            if not os.path.isdir(export_dir):
                try:
                    os.makedirs(config['export_dir'], exist_ok=True)  # Create the directory if it doesn't exist
                except Exception as e:
                    print(f"Error creating export directory '{config['export_dir']}': {e}")
                    return
            
            # Generate paths if the corresponding export option is true
            if export_options['csv']:
                config['csv'] = os.path.join(export_dir, "raw_data.csv")
                print(f"CSV export will be saved to: {config['csv']}")
            else:
                config['csv'] = None  # No CSV export

            if export_options['graph']:
                config['graph_out'] = os.path.join(export_dir, "final_comprehensive_plots.png")
                print(f"Graph export will be saved to: {config['graph_out']}")
            else:
                config['graph_out'] = None  # No graph export
            
            if export_options['animation']:
                config['animation_out'] = os.path.join(export_dir, "animation.mp4")

        else:
            config['csv'] = None  # No export
            config['graph_out'] = None  # No export

    else:
        config['csv'] = None  # Default to no CSV
        config['graph_out'] = None  # Default to no graph

    return config