# src/utils.py

import numpy as np  # For numerical operations and handling arrays
import matplotlib.pyplot as plt  # For plotting images and displaying results
import torch  # For PyTorch tensor operations and handling models
import torch.nn as nn  # For defining loss functions like MSELoss and building models

def show_processed_image(image_tensor):
    """
    Function to display a processed image.
    Args:
    - image_tensor (torch.Tensor): The image tensor after transformations.
    """
    # Convert the tensor from (C, H, W) to (H, W, C)
    image_np = image_tensor.permute(1, 2, 0).cpu().numpy()

    # Ensure the values are in the range [0, 1] for display
    image_np = np.clip(image_np, 0, 1)

    # Plot the image
    plt.imshow(image_np)
    plt.title('Processed Image')
    plt.axis('off')  # Hide axes
    plt.show()
    
# Custom split function
def split_dataset(dataset, test_size=0.2, random_state=42):
    # Split the indices of the dataset
    total_size = len(dataset)
    indices = np.arange(total_size)
    np.random.seed(random_state)
    np.random.shuffle(indices)
    
    # Calculate split sizes
    test_size = int(total_size * test_size)
    train_size = total_size - test_size
    
    # Split the indices into train and test
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    # Create subsets for training and validation
    train_subset = torch.utils.data.Subset(dataset, train_indices)
    val_subset = torch.utils.data.Subset(dataset, val_indices)
    
    return train_subset, val_subset

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    criterion = nn.MSELoss()  # Mean squared error loss
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0
        for right_eye_image, left_eye_image, labels, yaw_pitch_input in train_loader:
            optimizer.zero_grad()  # Zero the gradients
            outputs = model(right_eye_image, left_eye_image, yaw_pitch_input)  # Forward pass with yaw_pitch_input
            loss = criterion(outputs, labels)  # Calculate loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update model weights

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")

        # Validate the model
        validate_model(model, val_loader)

def validate_model(model, val_loader):
    criterion = nn.MSELoss()  # Mean squared error loss
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    with torch.no_grad():
        for right_eye_image, left_eye_image, labels, yaw_pitch_input in val_loader:  # Unpack 4 values
            outputs = model(right_eye_image, left_eye_image, yaw_pitch_input)  # Forward pass with yaw_pitch_input
            loss = criterion(outputs, labels)  # Calculate loss
            total_loss += loss.item()

    print(f"Validation Loss: {total_loss/len(val_loader)}")
