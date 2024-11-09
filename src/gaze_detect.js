const cv = require('opencv4nodejs');
const faceapi = require('face-api.js');
const path = require('path');
const { ChartJSNodeCanvas } = require('chartjs-node-canvas');
const fs = require('fs');
const { performance } = require('perf_hooks');

// Savitzky-Golay filter function
const applySavGolFilter = (data, windowLength = 9, polyorder = 2) => {
    if (data.length >= windowLength) {
        const sgf = require('scipy-signal');
        return sgf.savgol_filter(data, windowLength, polyorder);
    }
    return data;
};

// Function to process the image frame and extract gaze points
async function processFrame(frame, xData, yData, net, width, height) {
    const detections = await net.detect(frame);
    
    if (detections.length > 0) {
        const { alignedRect: { x, y, width: w, height: h } } = detections[0];
        const irisX = x + w / 2;
        const irisY = y + h / 2;

        // Store the iris coordinates
        xData.push(irisX);
        yData.push(irisY);

        // Draw circles on the frame to visualize iris points
        cv.drawContours(frame, [new cv.Rect(x, y, w, h)], 0, new cv.Vec(0, 255, 0), 2);
    }
    
    return frame;
}

// CLI argument parsing and setup
const args = process.argv.slice(2);
const source = args[0] || 'webcam';  // Default to 'webcam' if not provided
const pathToFile = args[1] || null;  // Optionally specify a path for image/video

// Initialize video capture
let cap;
if (source === 'webcam') {
    cap = new cv.VideoCapture(0);
} else if (source === 'image' || source === 'video') {
    if (!pathToFile) {
        console.log("You must provide a path to the image or video file.");
        process.exit(1);
    }
    cap = new cv.VideoCapture(pathToFile);
} else {
    console.log("Invalid source type provided.");
    process.exit(1);
}

// Load face-api.js model (on a Node.js environment, we use TensorFlow.js to load models)
async function loadFaceApiModels() {
    await faceapi.nets.ssdMobilenetv1.loadFromDisk(path.join(__dirname, 'models'));
    console.log("Models loaded.");
}

async function main() {
    // Initialize face-api.js and load the model
    await loadFaceApiModels();
    
    let xData = [];
    let yData = [];
    let timeData = [];
    
    const startTime = performance.now();

    // For webcam or video stream
    while (true) {
        let frame = cap.read();
        if (frame.empty) {
            break;
        }

        const currentTime = performance.now() - startTime;
        timeData.push(currentTime);

        // Process the frame to detect gaze
        frame = await processFrame(frame, xData, yData, faceapi, cap.get(cv.CAP_PROP_FRAME_WIDTH), cap.get(cv.CAP_PROP_FRAME_HEIGHT));

        // Apply Savitzky-Golay filter for smoothing
        const smoothedX = applySavGolFilter(xData);
        const smoothedY = applySavGolFilter(yData);

        // Display the processed frame
        cv.imshow('Gaze Tracking', frame);

        // Graphing (using Chart.js or similar)
        if (args.includes('--graph')) {
            const chartCallback = (chart) => {
                chart.render();
            };

            const chart = new ChartJSNodeCanvas({ width: 800, height: 600 });
            chart.render({
                type: 'line',
                data: {
                    labels: timeData,
                    datasets: [
                        {
                            label: 'X Coordinate',
                            data: smoothedX,
                            borderColor: 'red',
                            fill: false
                        },
                        {
                            label: 'Y Coordinate',
                            data: smoothedY,
                            borderColor: 'blue',
                            fill: false
                        }
                    ]
                },
                options: {
                    scales: {
                        x: { title: { display: true, text: 'Time (ms)' } },
                        y: { title: { display: true, text: 'Gaze Coordinates' } }
                    }
                }
            }).then(() => {
                // Save the plot as a PNG file or output as needed
                chart.toFile('gaze_plot.png');
            }).catch((err) => {
                console.error("Error creating chart:", err);
            });
        }

        // Exit if 'q' is pressed
        if (cv.waitKey(1) === 113) { // 'q' key
            break;
        }
    }

    // Cleanup
    cap.release();
    cv.destroyAllWindows();
}

main();
