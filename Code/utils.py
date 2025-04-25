import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import torch
from scipy.signal import savgol_filter
import numpy as np
import warnings

from config import angle_history , angle_threshold

warnings.filterwarnings("ignore")
    
# -------------------------
# Helper Functions for Preprocessing
# -------------------------
def get_img_tensor(pil_img, target_size, transform):
    iw, ih = pil_img.size
    if iw != target_size[0] or ih != target_size[1]:
        pil_img = pil_img.resize(target_size, Image.Resampling.BICUBIC)
    tensor_img = transform(pil_img)
    return torch.unsqueeze(tensor_img, 0)

def cut_resize_letterbox(image, bbox, target_size, margin=0.2):
    x, y, x2, y2 = bbox
    facebox_w = x2 - x
    facebox_h = y2 - y

    new_w = facebox_w * (1 + margin)
    new_h = facebox_h * (1 + margin)
    cx = x + facebox_w / 2
    cy = y + facebox_h / 2

    new_x = cx - new_w / 2
    new_y = cy - new_h / 2

    facebox_max_length = max(new_w, new_h)
    width_margin = (facebox_max_length - new_w) / 2
    height_margin = (facebox_max_length - new_h) / 2

    square_x = new_x - width_margin
    square_y = new_y - height_margin
    square_w = facebox_max_length

    iw, ih = image.size
    top = -square_y if square_y < 0 else 0
    left = -square_x if square_x < 0 else 0
    bottom = (square_y + square_w - ih) if (square_y + square_w) > ih else 0
    right = (square_x + square_w - iw) if (square_x + square_w) > iw else 0

    padded_width = int(iw + left + right)
    padded_height = int(ih + top + bottom)
    padded_img = Image.new('RGB', (padded_width, padded_height), (0, 0, 0))
    padded_img.paste(image, (int(left), int(top)))

    square_x = int(square_x + left)
    square_y = int(square_y + top)
    square_w = int(square_w)
    face_crop = padded_img.crop((square_x, square_y, square_x + square_w, square_y + square_w))
    face_letterbox = face_crop.resize(target_size, Image.Resampling.BICUBIC)
    scale_l = facebox_max_length / target_size[0]
    return face_letterbox, scale_l, square_x, square_y, square_w


# -------------------------
# Virtual Point Creation for Pose Estimation (68 Keypoints)
# -------------------------
def compute_virtual_nose_point(landmarks):
    """
    For 68-point model, the sorted keypoints are:
      - Jawline: indices 0-16 (with chin at index 8)
      - Nose tip: index 33
      - Eyes: indices 36-41 (right eye), 42-47 (left eye)
      - Mouth corners: indices 48 and 54

    This function fuses the nose tip, chin, and the mouth midpoint to yield a robust virtual nose.
    """
    chin = landmarks[8]
    nose_tip = landmarks[33]
    mouth_corner1 = landmarks[48]
    mouth_corner2 = landmarks[54]
    mouth_midpoint = (mouth_corner1 + mouth_corner2) / 2.0

    # Example weights – tune these based on empirical performance.
    weight_nose = 0.5
    weight_chin = 0.3
    weight_mouth = 0.2

    virtual_nose = (weight_nose * nose_tip + weight_chin * chin + weight_mouth * mouth_midpoint) / (weight_nose + weight_chin + weight_mouth)
    return virtual_nose

# -------------------------
# Angle Smoothing: Reduce Jitteriness
# -------------------------
def smooth_angles(current_angles):
    """ 
    Smooths angles using Savitzky-Golay filter and rejects sudden spikes. 
    Returns a stable version if jump exceeds allowed threshold.
    """
    global angle_history, angle_threshold

    if len(angle_history) == 0:
        angle_history.append(current_angles)
        return current_angles

    # Compare to the last angles
    last_angles = angle_history[-1]
    diffs = np.abs(np.array(current_angles) - np.array(last_angles))

    # If difference is too large, use last angles to ignore the spike
    if np.any(diffs > angle_threshold):
        current_angles = last_angles
    else:
        angle_history.append(current_angles)

    # Apply Savitzky-Golay smoothing if enough history is available
    if len(angle_history) >= 5:
        angle_array = np.array(angle_history)
        smoothed = savgol_filter(angle_array, window_length=5, polyorder=2, axis=0)
        return smoothed[-1]  # Return latest smoothed values

    return current_angles



def compute_head_pose_weighted_level68(landmarks, image_size, virtual_nose=None):
    """
    Enhanced head pose estimation with better yaw detection using additional landmarks.
    """

    if virtual_nose is None:
        virtual_nose = landmarks[30]  # Nose bridge tip (usually accurate for pitch)

    # Define corresponding 2D image points from detected facial landmarks
    image_points = np.array([
        landmarks[30],  # Nose tip
        landmarks[8],   # Chin
        landmarks[36],  # Left eye outer corner
        landmarks[45],  # Right eye outer corner
        landmarks[48],  # Left Mouth corner
        landmarks[54],  # Right Mouth corner
        landmarks[1],   # Jawline left (optional: adjust for accuracy)
        landmarks[15],  # Jawline right (optional: adjust for accuracy)
    ], dtype="double")

    # Define 3D model points (generic human face with more horizontal and vertical landmarks)
    model_points = np.array([
        [0.0, 0.0, 0.0],             # Nose tip
        [0.0, -63.6, -12.5],         # Chin
        [-42.0, 32.0, -26.0],        # Left eye outer corner
        [42.0, 32.0, -26.0],         # Right eye outer corner
        [-28.0, -28.9, -24.1],       # Left Mouth corner
        [28.0, -28.9, -24.1],        # Right Mouth corner
        [-50.0, 50.0, -50.0],        # Jawline left (optional: adjust for accuracy)
        [50.0, 50.0, -50.0],         # Jawline right (optional: adjust for accuracy)
    ], dtype="double")

    # Camera intrinsics
    focal_length = image_size[0]
    center = (image_size[0] / 2, image_size[1] / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")

    dist_coeffs = np.zeros((4, 1))  # No lens distortion

    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        return None, None, None, None, None

    # Convert rotation vector to rotation matrix
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    proj_matrix = np.hstack((rotation_matrix, translation_vector))
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)

    # Get yaw and roll from OpenCV (they are more stable)
    yaw, roll = euler_angles[1], euler_angles[2]

    # --- Custom Pitch Computation (from landmark alignment) ---
    # Use chin, nose, and vertical landmarks to refine pitch calculation
    nose = landmarks[30]
    chin = landmarks[8]
    left_eyebrow = landmarks[19]
    right_eyebrow = landmarks[24]

    # Vertical distance between chin and nose to calculate pitch
    dy = chin[1] - nose[1]
    dx = chin[0] - nose[0]
    pitch_radians = np.arctan2(dy, dx)  # Simplified angle from vertical line
    pitch_deg_custom = np.degrees(pitch_radians)

    # Use mid-eyebrow distance to adjust the pitch calculation (provides more accuracy)
    eyebrow_distance = abs(left_eyebrow[1] - right_eyebrow[1])  # Measure the vertical distance
    pitch_deg_custom += eyebrow_distance * 0.1  # Scale the eyebrow effect to refine pitch

    # Flip sign to match OpenCV pitch convention
    pitch_deg_custom = -pitch_deg_custom

    # Combine with OpenCV's pitch for comparison (optional)
    pitch_cv = euler_angles[0].item()
    pitch = 0.7 * pitch_cv + 0.3 * pitch_deg_custom  # Weighted combination

    # Smoothing
    smoothed = smooth_angles([pitch, yaw.item(), roll.item()])

    return smoothed[0], smoothed[1], smoothed[2], rotation_vector, translation_vector


# -------------------------
# Distance Estimation Based on Eye Distance
# -------------------------
def estimate_distance_from_eyes(left_eye, right_eye, predefined_eye_distance=5.0):
    """
    Estimate the distance between the user and the camera based on the horizontal distance
    between the left and right eye landmarks.
    :param left_eye: (x, y) coordinates of the left eye center
    :param right_eye: (x, y) coordinates of the right eye center
    :param predefined_eye_distance: Predefined distance (in cm) between the user and camera for calibration
    :return: Estimated distance in centimeters
    """
    # Calculate the Euclidean distance between the left and right eyes
    eye_distance_pixels = np.linalg.norm(np.array(left_eye) - np.array(right_eye))
    
    # Predefined real-world distance and corresponding eye distance in pixels at that distance
    predefined_eye_distance_pixels = 100  # This is an assumed pixel value at the predefined distance
    
    # Estimate distance based on eye distance
    estimated_distance = predefined_eye_distance * (predefined_eye_distance_pixels / eye_distance_pixels)
    
    return estimated_distance

# -------------------------
# Keypoints Drawing with Distance Estimation and Eye Cropping
# -------------------------

def draw_keypoints_on_image(pil_img, keypoints, x_offset=0, y_offset=0, scale=1.0, colors_list=None):
    draw = ImageDraw.Draw(pil_img)
    num_points = keypoints.shape[0]

    # Right and left eye landmarks
    right_eye_points = keypoints[36:41]  # Right eye points: indices 36 to 41
    left_eye_points = keypoints[42:47]   # Left eye points: indices 42 to 47

    # Compute shifted centers
    right_center_x = int(np.mean(right_eye_points[:, 0]))
    right_center_y = int(np.mean(right_eye_points[:, 1]))
    left_center_x = int(np.mean(left_eye_points[:, 0]))
    left_center_y = int(np.mean(left_eye_points[:, 1]))

    # Estimate distance from eyes
    estimated_distance = estimate_distance_from_eyes((left_center_x, left_center_y), (right_center_x, right_center_y))

    # Display distance on image
    font = ImageFont.load_default()
    text = f"Distance: {estimated_distance:.2f} cm"
    draw.text((10, 30), text, fill=(255, 255, 0), font=font)

    # Define the ROI size (Width and Height for rectangular bounding box)
    roi_w, roi_h = 60, 30
    frame_w, frame_h = pil_img.size

    # Right eye bounding box (Rectangle)
    r_x1 = max(0, right_center_x - roi_w // 2)
    r_y1 = max(0, right_center_y - roi_h // 2)
    r_x2 = min(frame_w, r_x1 + roi_w)
    r_y2 = min(frame_h, r_y1 + roi_h)

    # Left eye bounding box (Rectangle)
    l_x1 = max(0, left_center_x - roi_w // 2)
    l_y1 = max(0, left_center_y - roi_h // 2)
    l_x2 = min(frame_w, l_x1 + roi_w)
    l_y2 = min(frame_h, l_y1 + roi_h)
    
    # Crop the right and left eye regions
    right_eye_crop = pil_img.crop((r_x1, r_y1, r_x2, r_y2))
    left_eye_crop = pil_img.crop((l_x1, l_y1, l_x2, l_y2))

    # Draw rectangles around the eyes
    draw.rectangle([(r_x1, r_y1), (r_x2, r_y2)], outline="green", width=2)
    draw.rectangle([(l_x1, l_y1), (l_x2, l_y2)], outline="red", width=2)

    # Draw all keypoints
    for i in range(num_points):
        x = keypoints[i, 0] * scale + x_offset
        y = keypoints[i, 1] * scale + y_offset
        radius = 2

        # Debug color coding for eye regions
        if 36 <= i < 42:
            col = (0, 0, 255)  # Right eye
        elif 42 <= i < 48:
            col = (0, 255, 0)  # Left eye
        else:
            col = (255, 0 , 0 )

        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=col)

    return pil_img , right_eye_crop , left_eye_crop

# Assuming 'apply_hist_eq' function is already defined
def apply_hist_eq(image):
    """
    Apply Histogram Equalization to enhance the contrast.
    """
    # Convert PIL image to numpy array if it's in PIL format
    if isinstance(image, Image.Image):
        image = np.array(image)

    # Check if the image is in RGB format and convert to grayscale
    if len(image.shape) == 3:  # RGB format
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Apply Histogram Equalization
    enhanced_image = cv2.equalizeHist(image)

    # Convert grayscale to RGB by replicating the same grayscale values across the three channels
    enhanced_image_rgb = cv2.cvtColor(enhanced_image, cv2.COLOR_GRAY2RGB)

    # Convert the enhanced NumPy array back to PIL Image
    enhanced_image_pil = Image.fromarray(enhanced_image_rgb)

    return enhanced_image_pil

def get_img_tensor(pil_img, target_size, transform):
    iw, ih = pil_img.size
    if iw != target_size[0] or ih != target_size[1]:
        pil_img = pil_img.resize(target_size, Image.Resampling.BICUBIC)
    
    # Apply the transformations (ensure proper normalization and type conversion)
    tensor_img = transform(pil_img)
    return torch.unsqueeze(tensor_img, 0).float()  # Ensure to use float tensor
    
