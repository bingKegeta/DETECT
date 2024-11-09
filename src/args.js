const fs = require('fs');
const path = require('path');

// Parse arguments from command line
function parseArgs() {
    const args = process.argv.slice(2);
    const options = {
        source: null,
        path: null,
        graph: false,
        affine: false,
        csv: null,
        csv_interval: 1.0
    };

    // Map arguments to options
    for (let i = 0; i < args.length; i++) {
        switch (args[i]) {
            case '--source':
                options.source = args[++i];
                if (!['webcam', 'image', 'video'].includes(options.source)) {
                    throw new Error("Invalid source type provided. Must be 'webcam', 'image', or 'video'.");
                }
                break;
            case '--path':
                options.path = args[++i];
                break;
            case '--graph':
                options.graph = true;
                break;
            case '--affine':
                options.affine = true;
                break;
            case '--csv':
                options.csv = args[++i];
                break;
            case '--csv_interval':
                options.csv_interval = parseFloat(args[++i]);
                break;
            default:
                throw new Error(`Unknown argument: ${args[i]}`);
        }
    }

    // Validation for required fields
    if (!options.source) {
        throw new Error("The '--source' argument is required.");
    }
    if (options.source !== 'webcam' && !options.path) {
        throw new Error("The '--path' argument is required when 'source' is not 'webcam'.");
    }

    return options;
}

// Load configuration from a JSON file
function loadConfig(jsonFilePath) {
    if (!fs.existsSync(jsonFilePath)) {
        throw new Error(`The configuration file ${jsonFilePath} does not exist.`);
    }

    const config = JSON.parse(fs.readFileSync(jsonFilePath, 'utf-8'));

    // Validate required fields
    const requiredFields = ['source'];
    requiredFields.forEach(field => {
        if (!(field in config)) {
            throw new Error(`The configuration is missing the required '${field}' field.`);
        }
    });

    if (!['webcam', 'image', 'video'].includes(config.source)) {
        throw new Error("Invalid value for 'source'. Must be 'webcam', 'image', or 'video'.");
    }

    // Set default values
    config.path = config.path || null;
    config.graph = config.graph ?? true;
    config.affine = config.affine ?? false;
    config.csv_interval = config.csv_interval ?? 1.0;
    config.categorize = config.categorize ?? false;

    if (config.source !== 'webcam' && !config.path) {
        throw new Error("The 'path' field is required when 'source' is not 'webcam'.");
    }

    if (!('dot_display' in config)) {
        throw new Error("The 'dot_display' parameter (boolean) is not provided.");
    }

    // Handle 'export' sub-requirements
    if (config.export) {
        const exportOptions = config.export;
        if (typeof exportOptions !== 'object' || !('csv' in exportOptions) || !('graph' in exportOptions)) {
            throw new Error("The 'export' field must be an object containing 'csv' and 'graph' keys.");
        }

        if (exportOptions.csv || exportOptions.graph || exportOptions.animation) {
            if (!config.export_dir) {
                throw new Error("The 'export_dir' field is required when 'export.csv' or 'export.graph' is enabled.");
            }

            const exportDir = config.export_dir;
            if (!fs.existsSync(exportDir)) {
                fs.mkdirSync(exportDir, { recursive: true });
            }

            if (exportOptions.csv) {
                config.csv = path.join(exportDir, "raw_data.csv");
                console.log(`CSV export will be saved to: ${config.csv}`);
            } else {
                config.csv = null;
            }

            if (exportOptions.graph) {
                config.graph_out = path.join(exportDir, "final_comprehensive_plots.png");
                console.log(`Graph export will be saved to: ${config.graph_out}`);
            } else {
                config.graph_out = null;
            }

            if (exportOptions.animation) {
                config.animation_out = path.join(exportDir, "animation.mp4");
            }
        } else {
            config.csv = null;
            config.graph_out = null;
        }
    } else {
        config.csv = null;
        config.graph_out = null;
    }

    return config;
}

// Example usage
try {
    const options = parseArgs();
    console.log("Parsed command-line arguments:", options);

    const config = loadConfig('./config.json');  // Replace with actual path to JSON file
    console.log("Loaded configuration:", config);
} catch (error) {
    console.error(error.message);
}
