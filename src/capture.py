import cv2

def initialize_capture(source, path=None):
    if source == 'webcam':
        return cv2.VideoCapture(0)
    elif source in ['image', 'video']:
        if not path:
            raise ValueError("You must provide a --path to the image or video file.")
        return cv2.VideoCapture(path)
    else:
        raise ValueError("Invalid source type provided.")
