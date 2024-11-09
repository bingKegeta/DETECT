function initializeCapture(source, path = null) {
    if (source === 'webcam') {
        return new cv.VideoCapture(0);  // For the default webcam
    } else if (source === 'image' || source === 'video') {
        if (!path) {
            throw new Error("You must provide a --path to the image or video file.");
        }
        return new cv.VideoCapture(path);  // Open image or video file
    } else {
        throw new Error("Invalid source type provided.");
    }
}

