const cv = require('opencv4nodejs');
const faceapi = require('face-api.js');
const path = require('path');
const yargs = require('yargs');
const fs = require('fs');

// Initialize face-api.js
async function loadFaceApiModels() {
    await faceapi.nets.ssdMobilenetv1.loadFromDisk(path.join(__dirname, 'models'));
    await faceapi.nets.faceLandmark68Net.loadFromDisk(path.join(__dirname, 'models'));
    console.log("Models loaded.");
}

// Landmark indices for left and right irises (as in Python code)
const LEFT_IRIS = [474, 475, 476, 477];
const RIGHT_IRIS = [469, 470, 471, 472];
const PUPILS = [468, 473];

// Process each frame to detect face, iris, and draw landmarks
async function processFrame(frame, net) {
    const detections = await net.detect(frame);
    if (detections.length > 0) {
        detections.forEach((detection) => {
            const { alignedRect: { x, y, width, height } } = detection;
            const irisX = x + width / 2;
            const irisY = y + height / 2;

            // Draw face and iris landmarks
            frame.drawRectangle(new cv.Rect(x, y, width, height), new cv.Vec(0, 255, 0), 2);

            LEFT_IRIS.concat(RIGHT_IRIS, PUPILS).forEach((index) => {
                const landmark = detection.landmarks[index];
                const x = Math.round(landmark.x);
                const y = Math.round(landmark.y);
                frame.drawCircle(new cv.Point(x, y), 2, new cv.Vec(0, 0, 255), -1);
            });
        });
    }
    return frame;
}

// Webcam mode
function webcamMode(net) {
    const cap = new cv.VideoCapture(0);

    while (cap.isOpened()) {
        const frame = cap.read();
        if (frame.empty) {
            console.log("Ignoring empty frame.");
            continue;
        }

        processFrame(frame, net).then((processedFrame) => {
            cv.imshow('MediaPipe FaceMesh with Iris Tracking', processedFrame);
        });

        const key = cv.waitKey(1);
        if (key === 113) { // 'q' key
            break;
        }
    }

    cap.release();
    cv.destroyAllWindows();
}

// Image mode
function imageMode(imagePath, net) {
    const image = cv.imread(imagePath);
    if (!image) {
        console.log(`Error: Could not read image at ${imagePath}`);
        return;
    }

    processFrame(image, net).then((processedImage) => {
        cv.imshow('MediaPipe FaceMesh with Iris Tracking', processedImage);
        cv.waitKey(0);
        cv.destroyAllWindows();
    });
}

// Video mode
function videoMode(videoPath, net) {
    const capture = new cv.VideoCapture(videoPath);

    if (!capture.isOpened()) {
        console.log(`Error: Could not open video at ${videoPath}`);
        return;
    }

    while (true) {
        const frame = capture.read();
        if (frame.empty) {
            console.log("End of video or cannot fetch frame.");
            break;
        }

        processFrame(frame, net).then((processedFrame) => {
            cv.imshow('MediaPipe FaceMesh with Iris Tracking', processedFrame);
        });

        const key = cv.waitKey(1);
        if (key === 113) { // 'q' key
            break;
        }
    }

    capture.release();
    cv.destroyAllWindows();
}

// Command-line argument parsing
const argv = yargs
    .option('webcam', {
        alias: 'w',
        type: 'boolean',
        description: 'Use webcam for face tracking'
    })
    .option('image', {
        alias: 'i',
        type: 'string',
        description: 'Path to the image file'
    })
    .option('video', {
        alias: 'v',
        type: 'string',
        description: 'Path to the video file'
    })
    .demandOption(['webcam', 'image', 'video'], 'You must provide at least one mode (webcam, image, video)')
    .help()
    .argv;

// Initialize face-api.js and start the appropriate mode
async function main() {
    await loadFaceApiModels();

    const net = new faceapi.SsdMobilenetv1();

    if (argv.webcam) {
        webcamMode(net);
    } else if (argv.image) {
        imageMode(argv.image, net);
    } else if (argv.video) {
        videoMode(argv.video, net);
    }
}

main().catch((err) => {
    console.error('Error:', err);
});
