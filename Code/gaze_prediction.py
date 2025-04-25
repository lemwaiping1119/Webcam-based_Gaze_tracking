import tkinter as tk
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from config import right_eye_crop_array, left_eye_crop_array, yaw, pitch, gaze_history , y_shift
from utils import apply_hist_eq

import tkinter as tk
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
import threading

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
            offset_y = -200  # Example offset for the y-axis

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
