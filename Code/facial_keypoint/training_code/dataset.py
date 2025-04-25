import torch
from torch.utils.data import Dataset
import cv2
import torchvision.transforms as T
from training_code.keypoint_levels import get_keypoint_indices

class ProcessedFaceDataset(Dataset):
    def __init__(self, images, annotations, transform=None, level=1):
        """
        images: list of processed face images (numpy arrays in BGR format)
        annotations: list of corresponding keypoints (numpy arrays of shape (68,2)) or None
        transform: torchvision transform to apply on images; if None, a default transform is used.
        level: keypoint level (1 to 5) defining which subset of keypoints to use.
        """
        self.images = images
        self.annotations = annotations
        self.level = level
        self.kp_indices = get_keypoint_indices(level)
        self.num_keypoints = len(self.kp_indices)
        
        if transform is None:
            self.transform = T.Compose([
                T.ToTensor(),  # converts to tensor and scales pixel values to [0,1]
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = self.transform(img_rgb)
        kp = self.annotations[idx]
        if kp is not None:
            kp_subset = kp[self.kp_indices, :]
            kp_tensor = torch.tensor(kp_subset, dtype=torch.float32).view(-1)
        else:
            kp_tensor = torch.zeros(self.num_keypoints * 2, dtype=torch.float32)
        return img_tensor, kp_tensor
