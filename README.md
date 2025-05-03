# Webcam-Based Gaze Estimation System

This project presents a complete pipeline for webcam-based gaze estimation using facial landmark detection, head pose estimation, and convolutional neural networks (CNNs). It provides a user-friendly GUI for real-time calibration, gaze tracking, model training, and performance evaluation.

---

## 🧠 System Architecture

1. **Initialization**  
   The user launches the application and grants access to the webcam.

2. **Face & Eye Detection**  
   Facial landmarks are detected using MediaPipe, and both eyes are cropped using facial keypoints.

3. **Head Pose Estimation**  
   Yaw and pitch angles are calculated from facial landmarks to provide context for gaze prediction.

4. **Calibration Phase**  
   The user is shown a series of calibration dots on the screen.  
   At each point:
   - Eye images are captured
   - Head pose data (yaw, pitch) is recorded
   - Ground truth screen coordinates are stored

5. **Model Input**  
   The CNN receives:
   - Cropped left eye image  
   - Cropped right eye image  
   - Head pose angles (yaw, pitch)  
   and predicts the gaze coordinates (x, y).

6. **Evaluation**  
   Predicted gaze points are compared to ground truth using **Euclidean Distance (EUD)**.

7. **Visualization**  
   Graphs show:
   - Training and validation loss  
   - MAE, RMSE, R² metrics  
   - Gaze prediction error (EUD)

---

## 🎯 Features

- **Webcam Inference**: Real-time facial landmark and head pose detection.
- **Calibration**: Interactive calibration using multiple screen points.
- **Gaze Estimation**: Predicts user’s gaze position using CNN model.
- **Model Training**: Trains the CNN using the collected calibration data.
- **Result Visualization**: Shows loss curves, evaluation metrics, and EUD graph.
- **File Management**: Easily manage calibration datasets.
- **GUI**: Tkinter-based interface for easy interaction.

---

## 📦 Requirements

Ensure Python 3.6+ is installed. Required packages:

- `opencv-python`
- `torch`
- `torchvision`
- `Pillow`
- `numpy`
- `mediapipe`
- `tkinter`
- `threading`

Install them via:

```bash
pip install opencv-python torch torchvision pillow numpy mediapipe
```

---

## 🧍 Face and Keypoint Detection

The project uses `mp_detection` for face detection and a custom keypoint detector trained using the **300VW** dataset with **SimpleFaceNet**.

### 🏋️‍♂️ Training the Keypoint Detector

Navigate to the directory:

```bash
cd Code/facial_keypoint
```

Train the model:

```bash
python train.py
```

To visualize keypoints:

```bash
python train.py --visualize
```

---

## 🚀 How to Run the System

From the main directory:

```bash
cd Code
python main.py
```

### 👣 Workflow

1. Press **Start Webcam Inference** to begin face and eye detection.
2. Adjust the sliding window for eye bounding box if needed.
3. Click **Start Calibration** to collect gaze data. Make sure webcam inference is active.
4. Continue webcam inference to perform gaze estimation and evaluation.
5. Once all data is collected, **press 'q'** to exit webcam inference and **close the webcam window**.
6. Then click **Training** to train the model.
7. A graph will be generated to show the train and validation loss along with **MAE**, **RMSE**, and **R²**.
8. After training is finished, press **Start Gaze Estimation** to start model evaluation. A graph of **Euclidean Distance (EUD)** will be generated.

---

Make sure to keep the webcam active during calibration and evaluation. Exit the webcam window before starting training.


