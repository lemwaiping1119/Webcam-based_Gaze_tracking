# src/train.py

from model import AdvancedCNN  # Import the model architecture (from model.py)
from dataset import GazeEstimationDataset  # Import the dataset class (from dataset.py)
from utils import train_model, show_processed_image , split_dataset  # Import utility functions for training and image display
import torch.optim as optim  # For optimization algorithms (e.g., Adam)
import torch  # PyTorch library for tensor operations and model handling
import os  # For file handling (e.g., checking file existence and saving models)
from torch.utils.data import DataLoader  # For batching and loading datasets
import matplotlib.pyplot as plt  # For plotting images
import torchvision.transforms as transforms  # For image transformations (resizing, converting to tensor, etc.)
import torch.nn as nn  # For defining loss functions like MSELoss and building models



def train_gaze_estimation_model(csv_file,model_save_path , num_epochs=100, batch_size=32, learning_rate=1e-4):
    """
    Function to train the gaze estimation model.
    
    Args:
        csv_file (str): Path to the CSV file containing the calibration data.
        num_epochs (int): Number of epochs to train the model (default 100).
        batch_size (int): Batch size for training and validation (default 32).
        learning_rate (float): Learning rate for the optimizer (default 1e-4).
        model_save_path (str): Path to save the trained model (default "gaze_estimation_model.pth").
    """
    # Check if CSV file is None or empty
    if csv_file is None or not os.path.exists(csv_file) or os.path.getsize(csv_file) == 0:
        print(f"Error: The provided CSV file '{csv_file}' is either None, does not exist, or is empty.")
        return

    # Image transformations
    transform = transforms.Compose([ 
        transforms.Resize((30 , 60)),  # Resize images to a consistent size
        transforms.ToTensor(),  # Convert images to tensor format
    ])

    # Load the dataset
    dataset = GazeEstimationDataset(csv_file=csv_file, transform=transform , base_path= r'C:\Users\Lenovo\Desktop\FYP_LemWaiPing\Code' )
    
    # Fetch the first sample and show processed image
    right_eye_image, left_eye_image, _, _ = dataset[0]
    show_processed_image(right_eye_image)

    # Split the dataset into training and validation sets
    train_data, val_data = split_dataset(dataset)
    
    # Create DataLoader for training and validation
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    
    # Initialize the model
    model = AdvancedCNN(num_outputs=2)  # 2 outputs: x, y screen positions
    
    # Define the loss function and optimizer
    criterion = nn.MSELoss()  # Mean squared error loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Train the model
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=num_epochs)
    
    # Save the trained model
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

train_gaze_estimation_model(csv_file= r'calibration_results\calibration_data.csv', num_epochs=100, batch_size=32, learning_rate=1e-4, model_save_path="saved_model/gaze_estimation_model.pth")
