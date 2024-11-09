const { createCanvas } = require('canvas');
const fs = require('fs');
const path = require('path');
const { exec } = require('child_process');

class GazeAnimation {
    constructor(eyePositions, frameWidth, frameHeight, fps = 30, saveDir = 'frames') {
        this.eyePositions = eyePositions;
        this.fps = fps;
        this.currentFrame = 0;
        this.frameWidth = frameWidth;
        this.frameHeight = frameHeight;
        this.saveDir = saveDir;

        // Create directory for frames if it doesn't exist
        if (!fs.existsSync(saveDir)) {
            fs.mkdirSync(saveDir, { recursive: true });
        }

        // Setup canvas
        this.canvas = createCanvas(frameWidth, frameHeight);
        this.context = this.canvas.getContext('2d');

        // Calculate padding and eye dimensions
        const xValues = eyePositions.map(pos => pos[0]).filter(x => x !== null);
        const yValues = eyePositions.map(pos => pos[1]).filter(y => y !== null);

        const min_x = Math.min(...xValues);
        const max_x = Math.max(...xValues);
        const min_y = Math.min(...yValues);
        const max_y = Math.max(...yValues);

        const padding_x = (max_x - min_x) * 0.1 || 1;
        const padding_y = (max_y - min_y) * 0.1 || 1;

        this.ax_min_x = min_x - padding_x;
        this.ax_max_x = max_x + padding_x;
        this.ax_min_y = min_y - padding_y;
        this.ax_max_y = max_y + padding_y;

        // Artificial eye setup
        this.centerX = (this.ax_min_x + this.ax_max_x) / 2;
        this.centerY = (this.ax_min_y + this.ax_max_y) / 2;
        this.radius = Math.max((this.ax_max_x - this.ax_min_x), (this.ax_max_y - this.ax_min_y)) / 2 + Math.max(padding_x, padding_y);
    }

    drawEye() {
        // Draw the artificial eye
        this.context.beginPath();
        this.context.arc(this.centerX, this.centerY, this.radius, 0, Math.PI * 2);
        this.context.fillStyle = 'white';
        this.context.fill();
        this.context.lineWidth = 2;
        this.context.strokeStyle = 'black';
        this.context.stroke();
    }

    drawGazePoint(x, y) {
        // Draw gaze point
        this.context.beginPath();
        this.context.arc(x, y, 7.5, 0, Math.PI * 2);
        this.context.fillStyle = 'blue';
        this.context.fill();
    }

    updateFrame() {
        if (this.currentFrame < this.eyePositions.length) {
            const [x, y] = this.eyePositions[this.currentFrame];
            this.context.clearRect(0, 0, this.frameWidth, this.frameHeight);

            this.drawEye();

            if (x !== null && y !== null) {
                this.drawGazePoint(x, y);
            } else {
                this.drawGazePoint(this.frameWidth / 2, this.frameHeight / 2);
            }

            this.captureFrame();
            this.currentFrame += 1;

            // Schedule next frame
            setTimeout(() => this.updateFrame(), 1000 / this.fps);
        } else {
            console.log("Animation complete.");
        }
    }

    captureFrame() {
        // Save current frame as an image
        const framePath = path.join(this.saveDir, `frame_${String(this.currentFrame).padStart(4, '0')}.png`);
        const buffer = this.canvas.toBuffer('image/png');
        fs.writeFileSync(framePath, buffer);
        console.log(`Saved frame: ${framePath}`);
    }
}

// Main function to create the animation
function createGazeAnimation(eyePositions, frameWidth, frameHeight, fps = 30, savePath = 'animation.mp4') {
    const animation = new GazeAnimation(eyePositions, frameWidth, frameHeight, fps);
    animation.updateFrame();

    // Compile frames into video with FFmpeg
    const ffmpegCommand = `ffmpeg -r ${fps} -i ${animation.saveDir}/frame_%04d.png -vcodec libx264 -y ${savePath}`;
    console.log("Frames saved. To compile them into a video, run the following command:");
    console.log(ffmpegCommand);

    // Optionally, execute FFmpeg directly in Node.js
    exec(ffmpegCommand, (error, stdout, stderr) => {
        if (error) {
            console.error(`Error during video creation: ${error.message}`);
            return;
        }
        console.log(`Video saved to ${savePath}`);
    });
}

// Example usage
const exampleEyePositions = [
    [50, 50], [60, 60], [70, 80], [80, 90], [90, 100], // Add more coordinates as needed
];
createGazeAnimation(exampleEyePositions, 640, 480, 30);
