import os
import csv
import time
import threading
from datetime import datetime
import math

import cv2
import torch
import pygame
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as transforms
from tkinter import messagebox
import tkinter as tk

from utils import cut_resize_letterbox, get_img_tensor, compute_head_pose_weighted_level68, \
                  compute_virtual_nose_point, draw_keypoints_on_image, apply_hist_eq
from face_detection_setup import face_detection
from model import SimpleFaceNet, AdvancedCNN
from gaze_prediction import gaze_prediction_thread
from config import landmarks_history, calibration, right_eye_crop_array, left_eye_crop_array, \
                  is_webcam_inference_running, last_right_eye_image, last_left_eye_image, \
                  yaw, pitch, y_shift, global_x, global_y, calibration_data, colors, image_save_dir, \
                  calibration_data_list, gaze_history


#Approximate 1.30 minutes
def run_calibration_animation(
    screen_width=1920,
    screen_height=1080,
    rows=3,
    cols=3,
    margin=200,
    dot_radius=15,
    dot_color=(255, 0, 0),
    bg_color=(0, 0, 0),
    dot_speed=10,
    pause_time=1,
    threshold=0.002
):
    global global_x, global_y  # Declare the global variables

    pygame.init()
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Gaze Calibration with Sliding Dot Animation")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 48)

    # Generate grid calibration points with center first
    grid_spacing_x = (screen_width - 2 * margin) // (cols - 1)
    grid_spacing_y = (screen_height - 2 * margin) // (rows - 1)
    grid_points = [(margin + j * grid_spacing_x, margin + i * grid_spacing_y)
                   for i in range(rows) for j in range(cols)]
    center_point = (screen_width // 2, screen_height // 2)
    calibration_points = [center_point] + grid_points

    # Add corners of the screen to calibration points
    corner_offset = 10  # Space to add between the corner and the dot
    corner_points = [
        (corner_offset, corner_offset),  # Top-left corner
        (screen_width - corner_offset, corner_offset),  # Top-right corner
        (screen_width - corner_offset, screen_height - corner_offset),  # Bottom-right corner
        (corner_offset, screen_height - corner_offset),  # Bottom-left corner
        (corner_offset, corner_offset)  # Top-left corner (new corner)
        
    ]
    calibration_points += corner_points  # Append corners to the calibration points

    # Function to draw a shrinking and growing dot animation during the pause
    def animate_dot_transition(position):
        for r in range(dot_radius, 1, -1):
            screen.fill(bg_color)
            pygame.draw.circle(screen, dot_color, position, r)
            pygame.display.flip()
            clock.tick(60)
        time.sleep(0.1)  # Pause for a moment at smallest size
        for r in range(1, dot_radius + 1):
            screen.fill(bg_color)
            pygame.draw.circle(screen, dot_color, position, r)
            pygame.display.flip()
            clock.tick(60)

    # Function to move the dot smoothly
    def move_dot(current, target, speed):
        dx = target[0] - current[0]
        dy = target[1] - current[1]
        distance = math.hypot(dx, dy)
        if distance > speed:
            dx /= distance
            dy /= distance
            current[0] += dx * speed
            current[1] += dy * speed
        else:
            current[0], current[1] = target[0], target[1]
        return current

    # Function to handle text rendering and event checking
    def render_text_and_handle_events(text, countdown_start, position, screen, clock, bg_color):
        # Render the main text (centered slightly above the screen)
        text_surface = font.render(text, True, (255, 255, 255))  # White text
        text_rect = text_surface.get_rect(center=(position[0], position[1] - 100))  # Slightly above center
        screen.fill(bg_color)  # Clear the screen before drawing text
        screen.blit(text_surface, text_rect)

        # Render the countdown text (3, 2, 1) slightly below the screen center
        for i in range(countdown_start, 0, -1):
            countdown_text = font.render(str(i), True, (255, 255, 255))  # White text
            countdown_rect = countdown_text.get_rect(center=(position[0], position[1] + 100))  # Slightly below center
        
            # Clear previous countdown number by filling the screen with bg_color
            screen.fill(bg_color)  # Clear the screen before displaying the next number
        
            # Draw the main text and the countdown number
            screen.blit(text_surface, text_rect)
            screen.blit(countdown_text, countdown_rect)
        
            pygame.display.flip()
            time.sleep(1)  # Pause for 1 second between countdown numbers
    
        clock.tick(60)
        return True

    dot_pos = list(center_point)
    index = 0
    running = True
    last_norm = (None, None)

    # Define the directions including the diagonal ones
    directions = [
        (screen_width // 2, 100),  # Up
        (screen_width // 2, screen_height // 2),  # Return to center after Up
        (screen_width - 100, screen_height // 2),  # Right
        (screen_width // 2, screen_height // 2),  # Return to center after Right
        (screen_width // 2, screen_height - 100),  # Down
        (screen_width // 2, screen_height // 2),  # Return to center after Down
        (100, screen_height // 2),  # Left
        (screen_width // 2, screen_height // 2),  # Return to center after Left
        (screen_width - 100, 100),  # Up-Right
        (screen_width // 2, screen_height // 2),  # Return to center after Up-Right
        (screen_width - 100, screen_height - 100),  # Down-Right
        (screen_width // 2, screen_height // 2),  # Return to center after Down-Right
        (100, screen_height - 100),  # Down-Left
        (screen_width // 2, screen_height // 2),  # Return to center after Down-Left
        (100, 100),  # Up-Left
        (screen_width // 2, screen_height // 2),  # Return to center after Up-Left
    ]

    # Sequence 1: Calibration Point Animation
    while running and index < len(calibration_points):
        screen.fill(bg_color)
        if index == 0:
            running = render_text_and_handle_events("Look at the dot (don't move your head )", 3, dot_pos, screen, clock, bg_color)

        dot_pos = move_dot(dot_pos, calibration_points[index], dot_speed)
        pygame.draw.circle(screen, dot_color, (int(dot_pos[0]), int(dot_pos[1])), dot_radius)

        # Compute normalized position
        norm_x = dot_pos[0] / screen_width
        norm_y = dot_pos[1] / screen_height
        global_x, global_y = norm_x, norm_y  # Update global variables

        pygame.display.flip()

        if dot_pos == list(calibration_points[index]):
            start_time = time.time()
            while time.time() - start_time < pause_time:
                if not running:
                    break
            animate_dot_transition(calibration_points[index])
            index += 1

    # Sequence 2: Directional Movement Animation (Right -> Down -> Left -> Up)
    dot_pos = list(center_point)
    index = 0

    running = render_text_and_handle_events("Look at the dot (move ur head)", 3, dot_pos, screen, clock, bg_color)

    # Now proceed with directional movement loop
    while running and index < len(directions):  # Stop when direction_index exceeds 7
        screen.fill(bg_color)

        # Get the target position based on the current direction index
        target_pos = directions[index]

        # Move the dot toward the target position
        dot_pos = move_dot(dot_pos, target_pos, dot_speed)
        pygame.draw.circle(screen, dot_color, (int(dot_pos[0]), int(dot_pos[1])), dot_radius)
        
        # Compute normalized position
        norm_x = dot_pos[0] / screen_width
        norm_y = dot_pos[1] / screen_height
        global_x, global_y = norm_x, norm_y  # Update global variables

        pygame.display.flip()

        # Check if the dot has reached the target position
        if dot_pos == list(target_pos):
            # Wait for the pause_time after reaching the target
            start_time = time.time()
            while time.time() - start_time < pause_time:
                if not running:
                    break

            # Animate the dot transition (shrink and grow)
            animate_dot_transition(target_pos)

            # After the dot reaches the center, increment the direction_index to move to the next direction
            index += 1
            
            if index == len(directions):
                running = False

        clock.tick(60)

    pygame.quit()
    
    
# Function to save calibration data to CSV
def save_calibration_data_to_csv():
    fieldnames = ['right_eye_image', 'left_eye_image', 'yaw', 'roll', 'pitch', 'screen_position_x', 'screen_position_y']
    csv_file_path = os.path.join(image_save_dir, 'calibration_data.csv')

    # Check if the file exists, create a new one if not
    file_exists = os.path.isfile(csv_file_path)

    with open(csv_file_path, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        # Write header only if the file doesn't exist
        if not file_exists:
            writer.writeheader()

        # Write each calibration data entry as a row in the CSV
        for entry in calibration_data_list:
            writer.writerow(entry)
    
# Function for webcam inference and saving calibration data
def inference_webcam(loaded_model , landmark_transform, landmark_input_size=(224, 224)):
    global is_webcam_inference_running , landmarks_history, calibration_data, global_x, global_y, last_left_eye_image , last_right_eye_image , calibration , right_eye_crop_array , left_eye_crop_array , yaw , pitch 
    baseline_angles = None  # First detected face as baseline (0°)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to open webcam.")
        return

    print("Webcam opened. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        is_webcam_inference_running = True

        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        iw, ih = pil_img.size
        draw_img = pil_img.copy()
        draw = ImageDraw.Draw(draw_img)

        results = face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        boxes = []
        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                x1 = int(bboxC.xmin * iw)
                y1 = int(bboxC.ymin * ih)
                w = int(bboxC.width * iw)
                h = int(bboxC.height * ih)
                boxes.append((x1, y1, x1+w, y1+h))
                draw.rectangle([(x1, y1), (x1+w, y1+h)], outline="red", width=2)

        for bbox in boxes:
            face_crop, scale_l, x_offset, y_offset, crop_size = cut_resize_letterbox(pil_img, bbox, landmark_input_size, margin=0.5)
            draw.rectangle([(x_offset, y_offset), (x_offset+crop_size, y_offset+crop_size)], outline="blue", width=2)
            pfld_tensor = get_img_tensor(face_crop, landmark_input_size, landmark_transform)
            loaded_model.eval()
            with torch.no_grad():
                output = loaded_model(pfld_tensor)
            predicted_keypoints = output.cpu().numpy().flatten()
            landmarks = predicted_keypoints.reshape(-1, 2)
            landmarks = np.array([(pt[0] * scale_l + x_offset, pt[1] * scale_l + y_offset) for pt in landmarks])
            current_smoothed = np.mean(np.array(list(landmarks_history) + [landmarks]), axis=0)
            landmarks_history.append(landmarks)
            
            # Right and left eye landmarks
            right_eye_points = current_smoothed[36:41]  # Right eye points: indices 36 to 41
            left_eye_points = current_smoothed[42:47]   # Left eye points: indices 42 to 47

            # Shift eye landmarks downward
            right_eye_points_shifted = right_eye_points.copy()
            right_eye_points_shifted[:, 1] += float(y_shift)

            left_eye_points_shifted = left_eye_points.copy()
            left_eye_points_shifted[:, 1] += float(y_shift)

            # Update the smoothed landmarks with the shifted and scaled eye points
            current_smoothed[36:41] = right_eye_points_shifted
            current_smoothed[42:47] = left_eye_points_shifted

            virtual_nose = compute_virtual_nose_point(current_smoothed)

            result = compute_head_pose_weighted_level68(current_smoothed, (iw, ih), virtual_nose=virtual_nose)
            if result is not None:
                pitch, yaw, roll, rotation_vector, translation_vector = result
                if baseline_angles is None:
                    baseline_angles = (pitch, yaw, roll)
                rel_pitch = pitch - baseline_angles[0]
                rel_yaw = yaw - baseline_angles[1]
                rel_roll = roll - baseline_angles[2]

                try:
                    font = ImageFont.truetype("arial.ttf", 20)
                except IOError:
                    font = ImageFont.load_default()
                text = f"Rel: Pitch: {rel_pitch:.1f}°, Yaw: {rel_yaw:.1f}°, Roll: {rel_roll:.1f}°"
                draw.text((10, 10), text, fill=(255, 255, 0), font=font)
                #draw_3d_axis(draw, rotation_vector, translation_vector, (iw, ih), virtual_nose, axis_length=50)

            draw_img, right_eye_crop, left_eye_crop = draw_keypoints_on_image(draw_img, current_smoothed, 0, 0, scale=1.0, colors_list=colors)

            # Convert PIL images to NumPy arrays and acquire lock before updating the global arrays
            right_eye_crop_array = np.array(right_eye_crop)
            left_eye_crop_array = np.array(left_eye_crop)
                
            if calibration :
                # Save the cropped eye images to the disk with a timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                right_eye_path = os.path.join(image_save_dir, f"right_eye_crop_{timestamp}.png")
                left_eye_path = os.path.join(image_save_dir, f"left_eye_crop_{timestamp}.png")
    
                # Check if the new images are different from the last saved images
                if (right_eye_path != last_right_eye_image) and (left_eye_path != last_left_eye_image) and (global_x != 0) and (global_y != 0) :
                    # Save images using OpenCV (cv2.imwrite)
                    cv2.imwrite(right_eye_path, right_eye_crop_array)
                    cv2.imwrite(left_eye_path, left_eye_crop_array)
    
                    # Update last saved images
                    last_right_eye_image = right_eye_path
                    last_left_eye_image = left_eye_path
    
                    # Update calibration data with the image paths and pose information
                    calibration_data = {
                        'right_eye_image': right_eye_path,  # Path to the right eye crop image
                        'left_eye_image': left_eye_path,   # Path to the left eye crop image
                        'yaw': rel_yaw,
                        'roll': rel_roll,
                        'pitch': rel_pitch,
                        'screen_position_x': global_x,
                        'screen_position_y': global_y
                    }
    
                    # Add this calibration data entry to the list for later model training
                    calibration_data_list.append(calibration_data)
            

        # Display the frame with the annotations
        cv_img = cv2.cvtColor(np.asarray(draw_img), cv2.COLOR_RGB2BGR)
        cv2.imshow("Webcam Face & Landmark Detection", cv_img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    save_calibration_data_to_csv()
    is_webcam_inference_running = False

    cap.release()
    cv2.destroyAllWindows()
    
    
# Example function to handle gaze prediction with additional white lines across the entire screen
def gaze_prediction_thread(gaze_loaded_model):
    screen_width = 1980
    screen_height = 1080

    # Create the window once, outside the loop
    root = tk.Tk()
    root.title("Gaze Estimation Dot")
    root.geometry(f"{screen_width}x{screen_height}+0+0")  # Full screen size
    root.configure(bg='white')

    canvas = tk.Canvas(root, width=screen_width, height=screen_height, bg='black', bd=0, highlightthickness=0)
    canvas.pack()

    stop_thread = False  # This will be used to stop the loop gracefully

    def update_canvas():
        """ Function to update the canvas and check if we should stop the loop. """
        
        nonlocal stop_thread
        
        if stop_thread:
            root.quit()  # Quit the main loop from the main thread
            root.destroy()
            return  # Stop the update cycle

        # Check if eye crop arrays are available
        
        if right_eye_crop_array is None or left_eye_crop_array is None:
            print("Error: Eye crop arrays are empty.")
            stop_thread = True
            return

        try:
            # Convert the NumPy arrays to PIL images for processing
            right_eye_image_pil = Image.fromarray(right_eye_crop_array)
            left_eye_image_pil = Image.fromarray(left_eye_crop_array)

            # Apply Histogram Equalization to enhance pupil visibility
            right_eye_enhanced = apply_hist_eq(right_eye_image_pil)
            left_eye_enhanced = apply_hist_eq(left_eye_image_pil)

            # Ensure the pixel values are in the expected range (0-255 for uint8)
            right_eye_enhanced = np.clip(right_eye_enhanced, 0, 255).astype(np.uint8)
            left_eye_enhanced = np.clip(left_eye_enhanced, 0, 255).astype(np.uint8)

            # Convert to PIL Image for further processing
            right_eye_enhanced_rgb_pil = Image.fromarray(right_eye_enhanced)
            left_eye_enhanced_rgb_pil = Image.fromarray(left_eye_enhanced)

            # Image transformations: Resize to 30x60 and convert to tensor format
            transform = transforms.Compose([ 
                transforms.Resize((30 , 60)),  # Resize images to a consistent size
                transforms.ToTensor(),  # Convert images to tensor format
            ])

            # Apply transformation and add batch dimension
            right_eye_tensor = transform(right_eye_enhanced_rgb_pil).unsqueeze(0)  # Add batch dimension
            left_eye_tensor = transform(left_eye_enhanced_rgb_pil).unsqueeze(0)

            # Normalize yaw and pitch (Assuming yaw and pitch are available)
            normalized_yaw = yaw / 40.0  # Normalize yaw to the range [-1, 1]
            normalized_pitch = pitch / 20.0  # Normalize pitch to the range [-1, 1]

            # Create yaw and pitch tensor for input to the model
            yaw_pitch_input = torch.tensor([normalized_yaw, normalized_pitch], dtype=torch.float32).unsqueeze(0)

            # Forward pass through the gaze model
            with torch.no_grad():
                output = gaze_loaded_model(right_eye_tensor, left_eye_tensor, yaw_pitch_input)

            # Get predicted gaze positions (screen coordinates)
            screen_position_x, screen_position_y = output.squeeze(0).cpu().numpy()
            screen_position_x *= screen_width
            screen_position_y *= screen_height

            # Ensure the gaze position is within bounds
            screen_position_x = np.clip(screen_position_x, 0, screen_width)
            screen_position_y = np.clip(screen_position_y, 0, screen_height)

            # Add current gaze point to the history
            gaze_history.append((screen_position_x, screen_position_y))

            # Smooth the gaze by averaging the last few points in history
            if len(gaze_history) > 1:
                smoothed_gaze_x = np.mean([pt[0] for pt in gaze_history])
                smoothed_gaze_y = np.mean([pt[1] for pt in gaze_history])
            else:
                smoothed_gaze_x, smoothed_gaze_y = screen_position_x, screen_position_y

            # Define offsets
            offset_x = 0  # Example offset for the x-axis
            offset_y = 0  # Example offset for the y-axis

            # Apply offsets
            smoothed_gaze_x = smoothed_gaze_x + offset_x
            smoothed_gaze_y = smoothed_gaze_y + offset_y

            # Clear canvas and plot the gaze estimation dot at the new position
            canvas.delete("all")
            dot_radius = 10  # Radius of the dot
            canvas.create_oval(smoothed_gaze_x - dot_radius, smoothed_gaze_y - dot_radius,
                               smoothed_gaze_x + dot_radius, smoothed_gaze_y + dot_radius,
                               fill='red', outline='red')

        except Exception as e:
            print(f"Error in processing: {e}")

        # Schedule the next canvas update
        root.after(30, update_canvas)  # Update every 30ms

    # Start the update cycle
    update_canvas()

    root.mainloop()  # Keep the Tkinter window open until quit


def update_offset(value):
    global y_shift
    y_shift = value

# Function to start webcam inference with landmark model
def start_webcam_inference():
    """Start the webcam inference when the button is clicked."""
    try:
        # Define model path and parameters for landmark model
        landmark_model_path = r"facial_keypoint\saved_models\level_1\model_level_1_final.pth"
        landmark_input_size = (224, 224)
        landmark_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        
        # Force the model to use CPU
        device = torch.device("cpu")

        # Initialize the landmark model
        loaded_model = SimpleFaceNet(num_keypoints=68).to(device)

        # Check if the landmark model exists and load it
        if os.path.exists(landmark_model_path):
            loaded_model.load_state_dict(torch.load(landmark_model_path, map_location=device))
            print(f"Loaded landmark model from {landmark_model_path}")
        else:
            print(f"Pretrained landmark model not found at {landmark_model_path}.")
            messagebox.showerror("Model Not Found", f"Pretrained landmark model not found at {landmark_model_path}.")
            return

        # Start webcam inference with the loaded model
        inference_webcam(loaded_model, landmark_transform, landmark_input_size)

    except Exception as e:
        print(f"Error: {e}")
        messagebox.showerror("Error", f"An error occurred while starting webcam inference: {e}")

# Function to start gaze estimation model
def start_gaze_estimation():
    """Start the gaze estimation model."""
    
    if is_webcam_inference_running :
        try:
            # Define model path for gaze estimation
            gaze_model_path = r"gaze_estimation\saved_model\gaze_estimation_model.pth"

            # Force the model to use CPU
            device = torch.device("cpu")

            # Initialize the gaze model
            gaze_loaded_model = AdvancedCNN(num_outputs=2).to(device)

            # Check if the gaze model exists and load it
            if os.path.exists(gaze_model_path):
                gaze_loaded_model.load_state_dict(torch.load(gaze_model_path, map_location=device))
                print(f"Loaded gaze estimation model from {gaze_model_path}")
            else:
                print(f"Pretrained gaze estimation model not found at {gaze_model_path}. Proceeding without it.")

            # Optionally, you can add logic here to run the gaze model
            gaze_loaded_model.eval()
            gaze_prediction_thread(gaze_loaded_model)

        except Exception as e:
            print(f"Error: {e}")
            messagebox.showerror("Error", f"An error occurred while starting gaze estimation: {e}")
    else:
        print("Click the Start Webcam_inference buttom first")

def start_calibration_thread():
    
    if is_webcam_inference_running:
        """Start the calibration thread when the button is clicked."""
        try:
            # Start calibration animation in a separate thread
            calibration_thread = threading.Thread(target=run_calibration_animation_thread)
            calibration_thread.start()

        except Exception as e:
            print(f"Error: {e}")
            messagebox.showerror("Error", f"An error occurred while starting the calibration: {e}")
    else:
        print("Click the Start Webcam Inference first")
        
def run_calibration_animation_thread():
    global calibration_data , calibration
    calibration = True
    run_calibration_animation()
    


