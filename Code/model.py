# PyTorch and Torchvision for deep learning models and transformations
import torch
import torch.nn as nn
import torchvision.transforms as transforms


# ----- Define a simple CNN model for keypoint regression ----- 
# The model outputs 28 landmarks (56 values) in the order:
# - Jawline: indices 0-16
# - Right Eye: indices 17-20
# - Left Eye: indices 21-24
# - Nose Tip: index 25
# - Left Mouth Corner: index 26
# - Right Mouth Corner: index 27
class SimpleFaceNet(nn.Module):
    def __init__(self, num_keypoints=68):  # Adjust for 68 keypoints (136 values)
        super(SimpleFaceNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),   # 3 x 224 x 224
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),                              # 16 x 112 x 112
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),                              # 32 x 56 x 56
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)                               # 64 x 28 x 28
        )
        self.regressor = nn.Sequential(
            nn.Linear(64 * 28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, num_keypoints * 2)  # For 68 points, this should be 68 * 2 = 136 values.
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.regressor(x)
        return x
    
    
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

class SimpleCNN(nn.Module):
    def __init__(self, num_outputs=2):  # We predict screen_position_x and screen_position_y
        super(SimpleCNN, self).__init__()

        # Right eye pathway
        self.conv1_right = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2_right = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3_right = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # Left eye pathway
        self.conv1_left = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2_left = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3_left = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        # Calculate the size of the feature map after convolution
        self.feature_map_size = self._get_feature_map_size(30, 60)

        # Fully connected layers after concatenating features from both eyes and yaw/pitch
        self.fc1 = nn.Linear(self.feature_map_size * 2 + 2, 512)  # Adjusted for new feature map size + yaw/pitch input
        self.fc2 = nn.Linear(512, num_outputs)  # Output 2 values: x and y screen positions

    def _get_feature_map_size(self, height, width):
        # Pass a dummy tensor through the network to calculate feature map size
        dummy_input = torch.ones(1, 3, height, width)
        x = self.pool(torch.relu(self.conv1_right(dummy_input)))
        x = self.pool(torch.relu(self.conv2_right(x)))
        x = self.pool(torch.relu(self.conv3_right(x)))
        return x.numel()  # Return the number of elements in the feature map

    def forward(self, right_eye_input, left_eye_input, yaw_pitch_input):
        # Right eye pathway
        x_right = self.pool(torch.relu(self.conv1_right(right_eye_input)))
        x_right = self.pool(torch.relu(self.conv2_right(x_right)))
        x_right = self.pool(torch.relu(self.conv3_right(x_right)))

        # Left eye pathway
        x_left = self.pool(torch.relu(self.conv1_left(left_eye_input)))
        x_left = self.pool(torch.relu(self.conv2_left(x_left)))
        x_left = self.pool(torch.relu(self.conv3_left(x_left)))

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