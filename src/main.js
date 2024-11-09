const { app, BrowserWindow, ipcMain } = require('electron');
const cv = require('opencv4nodejs');
const path = require('path');
const faceapi = require('face-api.js');
const fs = require('fs');
const yargs = require('yargs');
const express = require('express');
const bodyParser = require('body-parser');
const { exec } = require('child_process');

// Load face-api.js models
async function loadFaceApiModels() {
    await faceapi.nets.ssdMobilenetv1.loadFromDisk(path.join(__dirname, 'models'));
    await faceapi.nets.faceLandmark68Net.loadFromDisk(path.join(__dirname, 'models'));
    console.log("Models loaded.");
}

// Initialize the webcam capture
function initWebcamCapture() {
    const cap = new cv.VideoCapture(0); // 0 for the default webcam

    if (!cap.isOpened()) {
        console.error("Error: Could not open webcam.");
        return null;
    }

    return cap;
}

// Process frame to detect iris and face
async function processFrame(frame, net) {
    const detections = await net.detect(frame);
    let irisDetected = false;
    if (detections.length > 0) {
        detections.forEach((detection) => {
            const { alignedRect: { x, y, width, height } } = detection;
            const irisX = x + width / 2;
            const irisY = y + height / 2;
            irisDetected = true;
            // Draw face rectangle
            frame.drawRectangle(new cv.Rect(x, y, width, height), new cv.Vec(0, 255, 0), 2);
            // Draw detected iris center
            frame.drawCircle(new cv.Point(irisX, irisY), 5, new cv.Vec(255, 0, 0), -1);
        });
    }
    return { frame, irisDetected };
}

// Create and configure the Electron window
function createWindow() {
    const win = new BrowserWindow({
        width: 1280,
        height: 720,
        webPreferences: {
            nodeIntegration: true
        }
    });

    win.loadURL(path.join(__dirname, 'index.html')); // Load the HTML file for the UI

    return win;
}

// Setup server for handling real-time data
function setupServer() {
    const app = express();
    app.use(bodyParser.json());

    app.post('/sendData', (req, res) => {
        console.log(req.body);
        res.status(200).send({ message: 'Data received' });
    });

    app.listen(3000, () => {
        console.log('Server running on http://localhost:3000');
    });
}

// Main function to start the application
async function main() {
    // Load models for face-api.js
    await loadFaceApiModels();

    const cap = initWebcamCapture();
    if (!cap) {
        console.error("Webcam initialization failed.");
        app.quit();
        return;
    }

    const net = new faceapi.SsdMobilenetv1(); // Initialize face-api.js model

    let paused = false; // Manage playback state
    const timeData = [];
    const xData = [];
    const yData = [];

    // Initialize Electron App
    app.whenReady().then(() => {
        const win = createWindow();

        let frameInterval = setInterval(async () => {
            if (!paused) {
                let frame = cap.read();
                if (frame.empty) {
                    console.log("End of video or empty frame.");
                    cap.release();
                    clearInterval(frameInterval);
                    return;
                }

                const { frame: processedFrame, irisDetected } = await processFrame(frame, net);
                
                if (irisDetected) {
                    const currentTime = Date.now();
                    timeData.push(currentTime);
                    // Update the data and UI (example: update the plot or send data to renderer)
                    win.webContents.send('updateData', { timeData, xData, yData });
                }

                // Display the processed frame in the Electron window
                win.webContents.send('updateFrame', processedFrame);
            }
        }, 1000 / 30); // Adjust the interval for frame rate (30 FPS)

        // Listen for pause/play toggle from renderer process
        ipcMain.on('togglePause', () => {
            paused = !paused;
            win.webContents.send('pauseState', paused);
        });

        // Listen for close event
        ipcMain.on('closeApp', () => {
            cap.release();
            app.quit();
        });
    });

    setupServer(); // Start a simple server for real-time data communication

    // Handle app quit
    app.on('window-all-closed', () => {
        if (process.platform !== 'darwin') app.quit();
    });
}

main().catch((err) => {
    console.error('Error:', err);
    app.quit();
});
