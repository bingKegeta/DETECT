import argparse

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
