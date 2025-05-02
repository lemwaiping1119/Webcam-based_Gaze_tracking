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
```
##Face and Keypoint Detection
This project uses mp_detection for face detection. The facial keypoint detector is trained using the 300VW dataset and the SimpleFaceNet architecture.

Training the Keypoint Detector
Go to the Code/facial_keypoint directory:

```bash
cd Code/facial_keypoint
```
Run the training script:


```bash
python train.py
```
If you want to visualize the keypoint positions on each photo, run:

```bash
python train.py --visualize
```
After training is complete, you can return to the Code directory and run the main interface.

# Starting the Calibration and Gaze Estimation Process
In the Code directory, run the main program:

```bash
python main.py
```
The interface will appear. Start the webcam inference by pressing the first button, Start Webcam Inference.
Optionally, adjust the sliding window for Start Gaze Estimation to cover the bounding box of your eye.
Press the second button, Start Calibration, to start capturing eye photos for gaze position estimation.

Next , click the next buttom 'Training' if u think the dataset is enough. Model will be trained and relative Graph will be generated.

Press the Start Gaze Estimation button to begin estimating gaze points.
A black screen will pop up. Look at the screen, and the gaze estimation model will predict the location of the dot on the screen based on your gaze.
Ensure your webcam is connected and functioning properly.


Calibration and gaze estimation may require several attempts to achieve optimal accuracy.
Make sure the y_offset is properly adjusted for your setup during calibration for more accurate results.






