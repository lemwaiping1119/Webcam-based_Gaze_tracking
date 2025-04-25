# Calibration and Webcam Interface

This project provides an interactive user interface for webcam-based calibration and gaze estimation. It uses computer vision models to detect faces, estimate head poses, and track eye movements using webcam inference. The interface also allows users to start calibration, adjust gaze estimation, and manage calibration data files.

## Features

- **Webcam Inference**: Start webcam inference for real-time face and landmark detection.
- **Calibration**: Perform calibration with the ability to adjust the `y_offset` for more accurate calibration.
- **Gaze Estimation**: Estimate gaze points based on the webcam input.
- **File Management**: Delete calibration files saved on the local machine.
- **Graphical User Interface**: A simple and intuitive GUI built with Tkinter.

## Requirements

Make sure you have Python 3.6 or higher installed, along with the following dependencies:

- `opencv-python`
- `torch`
- `PIL` (Pillow)
- `numpy`
- `torchvision`
- `mediapipe`
- `tkinter`
- `threading`

To install the required dependencies, you can use the following command:

```bash
pip install opencv-python torch torchvision pillow numpy mediapipe
