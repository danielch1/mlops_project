import torch
from PIL import Image
import os

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

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]

        return image, label
    
    def save(self, file_path):
        saved_data = {
            'file_paths': self.file_paths,
            'labels': self.labels,
        }
        torch.save(saved_data, file_path)

    def load(self, file_path):
        saved_data = torch.load(file_path)
        file_paths = saved_data['file_paths']
        labels = saved_data['labels']

        return file_paths, labels