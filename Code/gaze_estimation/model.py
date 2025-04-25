# src/model.py

import torch.nn as nn
import torch

class AdvancedCNN(nn.Module):
    def __init__(self, num_outputs=2):
        super(AdvancedCNN, self).__init__()

        # Right eye pathway
        self.conv1_right = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2_right = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3_right = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv4_right = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # Left eye pathway
        self.conv1_left = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2_left = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3_left = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv4_left = nn.Conv2d(256, 512, kernel_size=3, padding=1)

        # Calculate the size of the feature map after convolution
        self.feature_map_size = self._get_feature_map_size(30, 60)

        # Fully connected layers after concatenating features from both eyes and yaw/pitch
        self.fc1 = nn.Linear(self.feature_map_size * 2 + 2, 1024)  # Adjusted for new feature map size + yaw/pitch input
        self.fc2 = nn.Linear(1024, num_outputs)  # Output 2 values: x and y screen positions

    def _get_feature_map_size(self, height, width):
        # Pass a dummy tensor through the network to calculate feature map size
        dummy_input = torch.ones(1, 3, height, width)
        x = self.pool(torch.relu(self.conv1_right(dummy_input)))
        x = self.pool(torch.relu(self.conv2_right(x)))
        x = self.pool(torch.relu(self.conv3_right(x)))
        x = self.pool(torch.relu(self.conv4_right(x)))
        return x.numel()  # Return the number of elements in the feature map

    def forward(self, right_eye_input, left_eye_input, yaw_pitch_input):
        # Right eye pathway
        x_right = self.pool(torch.relu(self.conv1_right(right_eye_input)))
        x_right = self.pool(torch.relu(self.conv2_right(x_right)))
        x_right = self.pool(torch.relu(self.conv3_right(x_right)))
        x_right = self.pool(torch.relu(self.conv4_right(x_right)))

        # Left eye pathway
        x_left = self.pool(torch.relu(self.conv1_left(left_eye_input)))
        x_left = self.pool(torch.relu(self.conv2_left(x_left)))
        x_left = self.pool(torch.relu(self.conv3_left(x_left)))
        x_left = self.pool(torch.relu(self.conv4_left(x_left)))

        # Flatten both pathways
        x_right = x_right.view(x_right.size(0), -1)
        x_left = x_left.view(x_left.size(0), -1)

        # Concatenate features from both eyes
        x = torch.cat((x_right, x_left), dim=1)

        # Concatenate yaw/pitch with image features
        x = torch.cat((x, yaw_pitch_input), dim=1)

        # Fully connected layers
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)  # Output 2 values: x and y screen positions

        return x
