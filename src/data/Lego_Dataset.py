import os
import random
import torch
from torch.utils.data import Dataset
from PIL import Image


class Lego_Dataset(torch.utils.data.Dataset):
    def __init__(self, file_paths, path, labels, transform=None):
        """
        Args:
            file_paths (list): List of file paths for the images.
            labels (list): List of corresponding labels.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform
        self.path = path

    def __len__(self):
        return len(self.file_paths)


    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        image = Image.open(os.path.join(self.path,img_path)).convert("RGB")

        random.seed(42)
        torch.manual_seed(42)
        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]

        return image, label
