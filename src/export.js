const fs = require('fs');
const createCsvWriter = require('csv-writer').createObjectCsvWriter;

function exportCsv(xData, yData, timeData, outputFile) {
    if (xData.length === 0 || yData.length === 0 || timeData.length === 0) {
        return;  // Ensure there is data to export
    }

    const csvWriter = createCsvWriter({
        path: outputFile,
        header: [
            { id: 'time', title: 'Time' },
            { id: 'x', title: 'X Coordinate' },
            { id: 'y', title: 'Y Coordinate' }
        ]
    });

    // Prepare data for writing
    const records = timeData.map((time, index) => ({
        time: time,
        x: xData[index],
        y: yData[index]
    }));

    csvWriter.writeRecords(records)
        .then(() => console.log('CSV file successfully created:', outputFile))
        .catch(error => console.error('Error writing CSV file:', error));
}

// Example usage
const xData = [100, 150, 200];  // Example X coordinates
const yData = [200, 250, 300];  // Example Y coordinates
const timeData = [1, 2, 3];     // Example time data
const outputFile = 'output.csv';

exportCsv(xData, yData, timeData, outputFile);
