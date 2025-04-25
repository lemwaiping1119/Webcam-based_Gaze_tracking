# -------------------------
# Keypoints Drawing with Distance Estimation and Eye Cropping
# -------------------------
from PIL import ImageDraw , ImageFont
import numpy as np
from utils import estimate_distance_from_eyes

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