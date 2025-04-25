# src/dataset.py

import pandas as pd  # For reading the CSV file
import numpy as np  # For numerical operations, especially handling arrays
from torch.utils.data import Dataset  # For creating a custom Dataset class
from PIL import Image  # For loading and manipulating images
import torchvision.transforms as transforms  # For image transformations (e.g., resizing, converting to tensor)
import cv2  # For using OpenCV, especially for histogram equalization (cv2 functions)
import torch
import os

def apply_hist_eq(image):
    """
    Apply Histogram Equalization to enhance the contrast.
    """
    if isinstance(image, Image.Image):
        image = np.array(image)

    if len(image.shape) == 3:  # RGB format
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    enhanced_image = cv2.equalizeHist(image)
    enhanced_image_rgb = cv2.cvtColor(enhanced_image, cv2.COLOR_GRAY2RGB)

    return enhanced_image_rgb

class GazeEstimationDataset(Dataset):
    def __init__(self, csv_file, transform=None, base_path=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.base_path = base_path

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get file paths and labels
        right_eye_image_path = self.data.iloc[idx, 0]
        left_eye_image_path = self.data.iloc[idx, 1]
        yaw = self.data.iloc[idx, 2]
        roll = self.data.iloc[idx, 3]
        pitch = self.data.iloc[idx, 4]
        screen_position_x = self.data.iloc[idx, 5]
        screen_position_y = self.data.iloc[idx, 6]

        # Normalize yaw and pitch
        normalized_yaw = yaw / 40.0
        normalized_pitch = pitch / 20.0
        yaw_pitch_input = np.array([normalized_yaw, normalized_pitch], dtype=np.float32)

        # Build absolute paths if base_path is provided
        if self.base_path:
            right_eye_image_path = os.path.join(self.base_path, right_eye_image_path)
            left_eye_image_path = os.path.join(self.base_path, left_eye_image_path)

        # Load and process images
        right_eye_image = Image.open(right_eye_image_path)
        left_eye_image = Image.open(left_eye_image_path)
        
        right_eye_image = apply_hist_eq(right_eye_image)
        left_eye_image = apply_hist_eq(left_eye_image)
        
        right_eye_image = Image.fromarray(right_eye_image)
        left_eye_image = Image.fromarray(left_eye_image)

        # Transform images to tensor format
        if self.transform:
            right_eye_image = self.transform(right_eye_image)
            left_eye_image = self.transform(left_eye_image)

        # Labels for regression (screen positions)
        labels = np.array([screen_position_x, screen_position_y], dtype=np.float32)

        return right_eye_image, left_eye_image, torch.tensor(labels), torch.tensor(yaw_pitch_input)
