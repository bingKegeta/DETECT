const cv = require('opencv.js');  // OpenCV.js library
const { FaceMesh } = require('@mediapipe/face_mesh');  // Mediapipe JS library
const { HorizontalRegion, VerticalRegion } = require('./utils');  // Import custom functions

// Initialize MediaPipe for face and iris detection
const faceMesh = new FaceMesh({ maxNumFaces: 1, refineLandmarks: true });

// Euclidean distance calculation
function distance(landmark1, landmark2) {
    return Math.sqrt(
        Math.pow(landmark2.x - landmark1.x, 2) +
        Math.pow(landmark2.y - landmark1.y, 2) +
        Math.pow(landmark2.z - landmark1.z, 2)
    );
}

// Process the frame for face and iris detection
async function processFrame(frame, xData, yData, applyAffine, display, categorize) {
    const img_h = frame.height;
    const img_w = frame.width;
    const rgbFrame = cv.cvtColor(frame, cv.COLOR_BGR2RGB);
    
    // Detect faces and landmarks using MediaPipe
    const results = await faceMesh.send({ image: rgbFrame });
    if (results.multiFaceLandmarks.length > 0) {
        const faceLandmarks = results.multiFaceLandmarks[0];

        // Optionally apply affine transform
        if (applyAffine) {
            const affineMatrix = getAffineTransform(faceLandmarks, img_w, img_h);
            frame = cv.warpAffine(frame, affineMatrix, new cv.Size(img_w, img_h));
        }

        // Store the left eye landmarks
        const left = faceLandmarks[468];
        const leftXlft = faceLandmarks[33];
        const leftXrgt = faceLandmarks[133];

        // Store the right eye landmarks
        const right = faceLandmarks[473];
        const rightXlft = faceLandmarks[36];
        const rightXrgt = faceLandmarks[263];

        const nose = faceLandmarks[4];

        // Get iris coordinates
        const leftIrisX = left.x - leftXrgt.x;
        const leftIrisY = nose.y - left.y;
        const rightIrisX = right.x - rightXlft.x;
        const rightIrisY = nose.y - right.y;

        // Average iris coordinates
        const avgX = (leftIrisX + rightIrisX) / 2;
        const avgY = (leftIrisY + rightIrisY) / 2;

        // Store the iris coordinates
        xData.push(avgX);
        yData.push(avgY);

        // Visual confirmation: draw tracked iris on the frame
        if (display) {
            cv.circle(frame, new cv.Point(left.x * img_w, left.y * img_h), 2, new cv.Scalar(0, 255, 0), -1);
            cv.circle(frame, new cv.Point(right.x * img_w, right.y * img_h), 2, new cv.Scalar(0, 255, 0), -1);
        }

        if (categorize) {
            const region = `Gaze Direction: ${HorizontalRegion(avgX)}-${VerticalRegion(avgY)}`;
            cv.putText(frame, region, new cv.Point(30, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, new cv.Scalar(255, 0, 0), 2);
        }

        return { frame, success: true };
    }

    return { frame, success: false };
}

function getAffineTransform(landmarks, imgW, imgH) {
    const leftEye = new cv.Point(landmarks[33].x * imgW, landmarks[33].y * imgH);
    const rightEye = new cv.Point(landmarks[263].x * imgW, landmarks[263].y * imgH);
    const noseBridge = new cv.Point(landmarks[1].x * imgW, landmarks[1].y * imgH);

    const refLeftEye = new cv.Point(imgW * 0.3, imgH * 0.4);
    const refRightEye = new cv.Point(imgW * 0.7, imgH * 0.4);
    const refNoseBridge = new cv.Point(imgW * 0.5, imgH * 0.6);

    const srcPoints = [leftEye, rightEye, noseBridge];
    const dstPoints = [refLeftEye, refRightEye, refNoseBridge];

    return cv.getAffineTransform(srcPoints, dstPoints);
}

// Example usage
(async function() {
    const frame = cv.imread('input_image.jpg');  // Input image
    const xData = [];
    const yData = [];

    const { frame: processedFrame, success } = await processFrame(frame, xData, yData, true, true, true);
    if (success) {
        cv.imshow('output_canvas', processedFrame);  // Show the processed image
    }
})();
