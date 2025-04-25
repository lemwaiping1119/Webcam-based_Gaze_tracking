import os
import cv2
import torch
import argparse
from preprocessing.frame_processing import process_frame_using_annotation, visualize_processed_frame
from training_code.dataset import ProcessedFaceDataset
from training_code.model import SimpleFaceNet
from training_code.train_pipeline import train_model
from torch.utils.data import DataLoader, random_split
import numpy as np
from datetime import datetime

# -------------------------
# Function to parse command-line arguments
# -------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Process and visualize dataset with keypoints.")
    parser.add_argument(
        '--visualize', 
        action='store_true', 
        help="Set this flag to visualize keypoints for each frame in the dataset"
    )
    return parser.parse_args()

# -------------------------
# Main Processing Loop for Dataset and Training
# -------------------------

# Parse command-line arguments
args = parse_args()

base_path = r"C:\BIGGEST_EYE\EYE_tracking\face-mesh-generator\dataset\300VW"
SAMPLE_EVERY = 10
TARGET_SIZE = (224, 224)
exit_flag = False
all_processed_images = []
all_processed_annotations = []

# Define a list of colors (BGR format) for 68 keypoints 
colors = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), 
    (255, 255, 0), (255, 0, 255), (0, 255, 255), 
    (128, 0, 0), (0, 128, 0), (0, 0, 128),
    (192, 0, 0), (0, 192, 0), (0, 0, 192),
    (255, 128, 0), (128, 255, 0), (128, 0, 255),
    (64, 64, 255), (255, 64, 128), (64, 255, 128)
]

visualize_keypoints = args.visualize  # Set to True if '--visualize' is passed

# -------------------------
# Frame Processing Loop: Process Video Frames and Annotations
# -------------------------
for idx_folder in os.listdir(base_path):
    if exit_flag:
        break
    video_folder = os.path.join(base_path, idx_folder)
    video_path = os.path.join(video_folder, 'vid.avi')
    annots_folder = os.path.join(video_folder, 'annot')
    
    cap = cv2.VideoCapture(video_path)
    processed_images = []
    processed_annotations = []
    frame_idx = 1
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % SAMPLE_EVERY != 0:
            frame_idx += 1
            continue
        
        annot_filename = str(frame_idx).zfill(6) + ".pts"
        annots_path = os.path.join(annots_folder, annot_filename)
        if not os.path.exists(annots_path):
            break  # break inner loop if annotation file is missing
        
        proc_img, proc_annots = process_frame_using_annotation(frame, annots_path, target_size=TARGET_SIZE, margin=0.1)
        processed_images.append(proc_img)
        processed_annotations.append(proc_annots)
        
        # Visualize processed frame: Draw all 68 keypoints if the flag is True
        vis_proc = proc_img.copy()
        if visualize_keypoints and proc_annots is not None:
            vis_proc = visualize_processed_frame(proc_img, proc_annots, colors_list=colors)
            cv2.imshow("Processed Frame Visualization", vis_proc)
        
        if cv2.waitKey(30) & 0xFF == ord('q'):
            exit_flag = True
            break
        
        frame_idx += 1
    cap.release()
    
    all_processed_images.append(processed_images)
    all_processed_annotations.append(processed_annotations)
    
    if exit_flag:
        break

cv2.destroyAllWindows()

# -------------------------
# Train Model for Every Keypoint Level from 1 to 5
# -------------------------
# Flatten the list of processed images and annotations
flat_images = [img for sublist in all_processed_images for img in sublist]
flat_annotations = [ann for sublist in all_processed_annotations for ann in sublist]

# For Loop: Train every level from 1 to 5 with 50 epochs and checkpoint every 10 epochs
for level in range(1, 6):
    print(f"\nTraining for keypoint level {level}...")

    # Create a unique directory for this level
    level_save_dir = os.path.join("saved_models", f"level_{level}")
    os.makedirs(level_save_dir, exist_ok=True)

    # Create the dataset for the current level
    full_dataset = ProcessedFaceDataset(flat_images, flat_annotations, level=level)

    # Split the dataset into training (80%) and validation (20%)
    dataset_size = len(full_dataset)
    val_size = int(0.2 * dataset_size)
    train_size = dataset_size - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Create DataLoader for the training and validation sets
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    # Set the device (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the model
    model = SimpleFaceNet(num_keypoints=full_dataset.num_keypoints).to(device)

    print(f"Training model for level {level} with {full_dataset.num_keypoints} keypoints...")

    # Train the model using the training function
    trained_model = train_model(model, train_loader, val_loader, num_epochs=50, learning_rate=1e-3, device=device, save_interval=10, save_dir=level_save_dir)

    # Save the final model for this level
    final_save_path = os.path.join(level_save_dir, f"model_level_{level}_final.pth")
    torch.save(trained_model.state_dict(), final_save_path)
    print(f"Final model for level {level} saved to {final_save_path}")
