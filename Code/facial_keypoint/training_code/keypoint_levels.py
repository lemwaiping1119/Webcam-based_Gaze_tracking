def get_keypoint_indices(level):
    """
    Returns a sorted list of keypoint indices (from the full 68)
    based on the desired level.

    Level definitions for robust head pose and eye tracking:
      Level 1: Full (68 points) - All keypoints.
      Level 2: Detailed (40 points) - Jawline, nose, eyes, mouth corners.
      Level 3: Intermediate (50 points) - Jawline, eyebrows, nose, eyes, mouth corners.
      Level 4: Simplified (32 points) - Jawline, eyes, nose tip, mouth corners.
      Level 5: Minimal (22 points) - Jawline, nose tip, mouth corners, and representative eye landmarks.
    """
    if level == 1:
        indices = list(range(68))
    elif level == 2:
        indices = list(range(0, 17)) + list(range(27, 36)) + list(range(36, 48)) + [48, 54]
    elif level == 3:
        indices = list(range(0, 17)) + list(range(17, 27)) + list(range(27, 36)) + list(range(36, 48)) + [48, 54]
    elif level == 4:
        indices = list(range(0, 17)) + list(range(36, 48)) + [33, 48, 54]
    elif level == 5:
        indices = list(range(0, 17)) + [33, 48, 54, 39, 45]
    else:
        raise ValueError("Level must be between 1 and 5")
    return sorted(list(set(indices)))
