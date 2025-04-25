# -------------------------
# Helper Functions for Preprocessing
# -------------------------
from PIL import Image
import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
import torch
from PIL import Image
from scipy.signal import savgol_filter 

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
