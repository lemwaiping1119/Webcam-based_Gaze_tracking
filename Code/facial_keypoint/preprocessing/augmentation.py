import cv2
import random
import numpy as np

def random_horizontal_flip(image, keypoints, p=0.5, flip_pairs=None):
    """
    Randomly flips the image horizontally with probability p.
    Adjusts keypoints accordingly and swaps symmetric pairs if provided.
    Returns the (possibly flipped) image, adjusted keypoints, and a flag indicating a flip.
    """
    flipped = False
    if keypoints is not None and random.random() < p:
        flipped = True
        # Flip image horizontally
        image = cv2.flip(image, 1)
        w = image.shape[1]
        # Mirror the x coordinates
        keypoints[:, 0] = w - keypoints[:, 0]
        # Swap symmetric keypoints if mapping provided
        if flip_pairs is not None:
            kp_copy = keypoints.copy()
            for i, j in flip_pairs:
                keypoints[i, :], keypoints[j, :] = kp_copy[j, :], kp_copy[i, :]
    return image, keypoints, flipped

def random_rotation(image, keypoints, angle_range=(-15, 15)):
    """
    Rotates the image and keypoints by a random angle within the specified range.
    Returns the rotated image and adjusted keypoints.
    """
    angle = random.uniform(*angle_range)
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, rot_matrix, (w, h))
    rotated_kp = None
    if keypoints is not None:
        ones = np.ones((keypoints.shape[0], 1))
        kp_homog = np.hstack([keypoints, ones])
        rotated_kp = np.dot(rot_matrix, kp_homog.T).T
    return rotated_image, rotated_kp
