import cv2
import os
import numpy as np
from preprocessing.augmentation import random_horizontal_flip, random_rotation
from preprocessing.utils import load_pts

def process_frame_using_annotation(frame, annots_path, target_size=(224,224), margin=0.1):
    """
    Processes a frame by:
      - Loading the annotation (.pts) file containing 68 keypoints.
      - Computing a bounding box around the landmarks and adding a margin.
      - Cropping and resizing the face.
      - Adjusting keypoint coordinates.
      - Applying data augmentations (horizontal flip with proper symmetric swaps and rotation).
      
    Returns:
      resized_face: The processed face image (BGR format).
      annots_adjusted: The adjusted 68 keypoints in the processed image.
    """
    if os.path.exists(annots_path):
        annots = load_pts(annots_path)
        annots = np.array(annots)
    else:
        annots = None
    
    if annots is not None:
        min_x = int(np.min(annots[:, 0]))
        max_x = int(np.max(annots[:, 0]))
        min_y = int(np.min(annots[:, 1]))
        max_y = int(np.max(annots[:, 1]))
        box_w = max_x - min_x
        box_h = max_y - min_y
        new_margin = margin + 0.1
        delta_x = int(box_w * new_margin)
        delta_y = int(box_h * new_margin)
        x1 = max(0, min_x - delta_x)
        y1 = max(0, min_y - delta_y)
        x2 = min(frame.shape[1], max_x + delta_x)
        y2 = min(frame.shape[0], max_y + delta_y)
        cropped_face = frame[y1:y2, x1:x2]
        annots_adjusted = annots.copy()
        annots_adjusted[:, 0] -= x1
        annots_adjusted[:, 1] -= y1
    else:
        cropped_face = frame
        annots_adjusted = None

    orig_h, orig_w = cropped_face.shape[:2]
    resized_face = cv2.resize(cropped_face, target_size, interpolation=cv2.INTER_AREA)
    scale_x = target_size[0] / orig_w
    scale_y = target_size[1] / orig_h

    if annots_adjusted is not None:
        annots_adjusted[:, 0] *= scale_x
        annots_adjusted[:, 1] *= scale_y
    else:
        annots_adjusted = None

    # Define symmetric flip pairs for 68 keypoints:
    flip_pairs_68 = [
        (0, 16), (1, 15), (2, 14), (3, 13), (4, 12), (5, 11), (6, 10), (7, 9),(31,35),
        # index 8 remains unchanged (chin center)
        (17, 26), (18, 25), (19, 24), (20, 23), (21, 22),
        (36, 45), (37, 44), (38, 43), (39, 42), (40, 47), (41, 46),
        (48, 54), (49, 53), (50, 52), (55, 59), (56, 58),
        (60, 64), (61, 63), (65, 67)  # 62 and 66 remain unchanged (for inner lip)
    ]

    # Data augmentation: horizontal flip and rotation
    if annots_adjusted is not None:
        resized_face, annots_adjusted, flipped_flag = random_horizontal_flip(
            resized_face, annots_adjusted, p=0.5, flip_pairs=flip_pairs_68)
        resized_face, annots_adjusted = random_rotation(resized_face, annots_adjusted, angle_range=(-15,15))

    resized_face = cv2.resize(resized_face, target_size, interpolation=cv2.INTER_AREA)

    # Now, assuming the .pts file provided 68 points, simply copy the adjusted points
    if annots_adjusted is not None and annots_adjusted.shape[0] >= 68:
        new_annots = annots_adjusted.copy()
        annots_adjusted = new_annots

    return resized_face, annots_adjusted

# -------------------------
# Visualization: Draw 68 Keypoints
# -------------------------
def visualize_processed_frame(proc_img, proc_annots, colors_list):
    """
    Visualizes the processed frame by drawing all 68 keypoints.
    
    Parameters:
      proc_img: Processed face image (BGR format)
      proc_annots: 68 keypoints (numpy array of shape (68,2))
      colors_list: List of colors for keypoints.
    
    Returns:
      vis_img: The visualization image with keypoints drawn.
    """
    vis_img = proc_img.copy()
    for idx, annot in enumerate(proc_annots):
        vis_img = cv2.circle(vis_img, tuple(annot.astype(int)), 2, colors_list[idx % len(colors_list)], -1)
    return vis_img
