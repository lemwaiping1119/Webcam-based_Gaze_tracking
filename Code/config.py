import os
import torchvision.transforms as transforms
import warnings
from collections import deque

warnings.filterwarnings("ignore")


import pygame
import time
import math

# -------------------------
# Global Variables
# -------------------------
angle_history = deque(maxlen=10)  # Store the last 10 angles for smoothing
angle_threshold = 400  # degrees - max allowed frame-to-frame change
landmarks_history = deque(maxlen=5)  # Store the last 10 landmarks for smoothing
gaze_history = deque(maxlen=30)


# Image transformations
transform = transforms.Compose([ 
    transforms.Resize((30, 60)),
    transforms.ToTensor(),
])

# Directory to save images
image_save_dir = r'gaze_estimation\calibration_results'

if not os.path.exists(image_save_dir):
    os.makedirs(image_save_dir)

colors = [(255, 0, 0)] * 32


# -------------------------
# Global Variables for Inference and Calibration
# -------------------------
global calibration, right_eye_crop_array, left_eye_crop_array, is_webcam_inference_running
global last_right_eye_image, last_left_eye_image, yaw, pitch, y_shift, eye_size
global global_x, global_y, calibration_data, calibration_data_list

calibration = False
right_eye_crop_array = None
left_eye_crop_array = None
is_webcam_inference_running = False
last_right_eye_image = ""
last_left_eye_image = ""
yaw = None
pitch = None
y_shift = 0.0 
eye_size = 0.0
global_x = 0.0
global_y = 0.0

# Calibration Data Structure
calibration_data = {
    'right_eye_image': None,  
    'left_eye_image': None,   
    'yaw': None,             
    'roll': None,            
    'pitch': None,           
    'screen_position_x': None,
    'screen_position_y': None
}

calibration_data_list = []