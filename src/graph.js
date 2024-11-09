const Plotly = require('plotly.js-dist');
const fs = require('fs');
const smooth = require('smooth');

// Smoothing parameters
const WINDOW_LENGTH = 13;
const POLYORDER = 2;

class GazeTrackingPlotter {
    constructor() {
        this.initPlots();
    }

    initPlots() {
        // Initialize X and Y line plots
        this.plotDivX = document.createElement('div');
        this.plotDivY = document.createElement('div');
        document.body.appendChild(this.plotDivX);
        document.body.appendChild(this.plotDivY);

        // Initialize 2D scatter plot
        this.plotDiv2D = document.createElement('div');
        document.body.appendChild(this.plotDiv2D);

        // Initialize 3D scatter plot
        this.plotDiv3D = document.createElement('div');
        document.body.appendChild(this.plotDiv3D);
    }

    updatePlots(timeData, xData, yData) {
        // Apply smoothing (savgol-like filter)
        const xDataSmoothed = smooth(xData, { method: 'savgol', window: WINDOW_LENGTH, polynomial: POLYORDER });
        const yDataSmoothed = smooth(yData, { method: 'savgol', window: WINDOW_LENGTH, polynomial: POLYORDER });

        // Plot X Coordinate Over Time
        Plotly.newPlot(this.plotDivX, [{
            x: timeData,
            y: xDataSmoothed,
            mode: 'lines',
            name: 'X Coordinate',
            line: { color: 'red' }
        }], { title: 'X Coordinate Over Time' });

        // Plot Y Coordinate Over Time
        Plotly.newPlot(this.plotDivY, [{
            x: timeData,
            y: yDataSmoothed,
            mode: 'lines',
            name: 'Y Coordinate',
            line: { color: 'blue' }
        }], { title: 'Y Coordinate Over Time' });

        // Plot 2D Gaze Points Over Time
        Plotly.newPlot(this.plotDiv2D, [{
            x: xData,
            y: yData,
            mode: 'markers',
            marker: { color: 'rgba(255, 0, 0, 0.8)' }
        }], { title: '2D Gaze Points Over Time', xaxis: { title: 'X' }, yaxis: { title: 'Y' } });

        // Plot 3D Gaze Points
        Plotly.newPlot(this.plotDiv3D, [{
            x: xData,
            y: yData,
            z: timeData,
            mode: 'markers',
            marker: { size: 5, color: 'rgba(255, 0, 0, 0.8)' },
            type: 'scatter3d'
        }], { title: '3D Gaze Points' });
    }
}

function plotGazeTracking(timeData, xData, yData) {
    const plotter = new GazeTrackingPlotter();
    plotter.updatePlots(timeData, xData, yData);
}

function plotFinalGraphs(timeData, xData, yData, savePath = 'final_comprehensive_plots.html') {
    const plotter = new GazeTrackingPlotter();
    plotter.updatePlots(timeData, xData, yData);

    // Save final plot as HTML for viewing in a browser
    const htmlContent = `
        <html>
            <head>
                <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            </head>
            <body>
                ${plotter.plotDivX.outerHTML}
                ${plotter.plotDivY.outerHTML}
                ${plotter.plotDiv2D.outerHTML}
                ${plotter.plotDiv3D.outerHTML}
            </body>
        </html>`;
    fs.writeFileSync(savePath, htmlContent);
    console.log(`Final comprehensive plots saved as ${savePath}.`);
}

// Example usage:
const timeData = [0, 1, 2, 3, 4];  // Example time data
const xData = [100, 150, 200, 250, 300];  // Example X coordinates
const yData = [200, 250, 300, 350, 400];  // Example Y coordinates

plotGazeTracking(timeData, xData, yData);
plotFinalGraphs(timeData, xData, yData);
