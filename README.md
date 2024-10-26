> [!IMPORTANT]
> **MIGRATION TO JSON COMPLETED!!!**
> 
> This version removes the cli arguments previously required to run various configs in favor of using a JSON config file. 

> [!NOTE]
> It is now possible to export the raw coordinates of the eyes as a csv file, a png with 4 graphs showing more insight into the gathered data and a nifty mp4 animation showcasing an estimated simulation of the gaze through the gathered data!

# DETECT: Deception Tracking Through Eye Cues Technology

> Note: This project has just been started and is mostly research as of now.

Welcome to DETECT! This project aims to revolutionize the field of deception detection by leveraging advanced gaze tracking technology. Utilizing the powerful MediaPipe framework, DETECT analyzes eye cues to offer a new dimension in understanding and interpreting human behavior.

## 📜 Overview

DETECT (Deception Tracking Through Eye Cues Technology) is a cutting-edge project designed to track and analyze eye movements to assist in identifying potential deception. By triangulating the position of the iris, DETECT provides valuable insights into gaze patterns that can be indicative of truthfulness or deception.

<!-- ## 🚀 Features

- **Real-Time Gaze Tracking:** Accurate triangulation of the iris location for precise gaze direction analysis.
- **MediaPipe Integration:** Harnesses the power of MediaPipe for efficient and reliable eye cue extraction.
- **Deception Insights:** Provides a foundation for further research into gaze patterns and their correlation with deceptive behavior. -->
## ⚙️ Features

The following features are currently available (almost all are experimental :P):
|Feature|Description|
|-------|-----------|
|```affine```|Use the Face normalization algorithm for possible improvement in precision|
|```graph```|Graph the x-time and y-time plots to see the changes in real time|
|```dot_display```|Show the iris/pupil as tracked by mediapipe (might reduce load)|
|```export::csv```|Export the tracked eye data into a csv file for advanced analysis|
|```csv_interval [sec]```|🚨 **[Doesn't work]** Sets the time interval between each collected data point for the csv|
|```export::graph```|Export the tracked eye data and graph it for easier comprehension and basic analysis|
|```export::animation```|Estimate the gaze direction over time based on the tracked eye data|
___
More features will be coming soon...

## 🛠️ Installation

To get started with DETECT, follow these steps:

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/bingKegeta/DETECT.git
   cd DETECT
   ```

2. **Set Up a Virtual Environment:**
    - Using conda (recommended):
    ```bash
    conda create --name detect
    conda activate detect
    ```
   - Using venv:
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows, use `env\Scripts\activate`
   ```
   - Docker: 👨‍🍳🍳
   - Others: Task will be left to the reader

3. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

## 🔧 Usage

1. **Make a configuration file (JSON):**
   - The file contains various options but the basic schema is this:
   ```json
   {
      "source":["webcam", "image", "video"], # Choose any one of those
      "path": "/path/to/media", # Only include when the source is not webcam
      "graph": boolean, # Only if you want graph updates (fps will be hit)
      "affine": boolean, # If you want affine tranform processing
      "csv_interval": number, # Doesn't work yet
      "export": {
         "csv": boolean, # Do you want csv output?
         "graph": boolean, # Do you want a comprehensive graph?
         "animation": boolean, # Do you want a nifty animation?
      },
      "export_dir": "/path/to/dir/" # Where do you want to store those files?
   }
   ```
2. **Run the Application:**
   ```bash
   python main.py config.json
   ```

3. **Start Analyzing:**
   - The application will initiate your camera and begin tracking eye movements.
   - Use the provided interface to view and analyze gaze data.

<!-- ## 📚 Documentation

For detailed documentation and usage instructions, please refer to the [Wiki](https://github.com/bingKegeta/DETECT/wiki) or the `docs` directory. -->

<!-- ## 🎯 Contributing

We welcome contributions to enhance DETECT's capabilities! If you have ideas, bug reports, or wish to contribute, please follow these steps:

1. **Fork the Repository**
2. **Create a New Branch**
3. **Make Your Changes**
4. **Submit a Pull Request**

Please review our [Contribution Guidelines](CONTRIBUTING.md) before getting started. -->

<!-- ## 💬 Contact

For questions or support, feel free to reach out to us:

- **Email:** <email>
- **Issues:** [GitHub Issues](https://github.com/bingKegeta/DETECT/issues) -->

<!-- ## 🔗 Links

- [GitHub Repository](https://github.com/yourusername/DETECT)
- [Project Wiki](https://github.com/yourusername/DETECT/wiki)
- [Documentation](docs/) -->

## 📜 License

This project is licensed under the [MIT License](LICENSE).

---
<!-- 
Thank you for your interest in DETECT! We look forward to your contributions and hope you find our technology useful in advancing the study of human behavior. Happy detecting!

--- -->