# -*- coding: utf-8 -*-
import logging
import pandas as pd
import numpy as np
import torch
import os
import random
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms


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


if __name__ == '__main__':

    wd = os.getcwd()
    path = os.path.join(wd,"C:/Users/dchro/Documents/MLOps/mlops_project/data/external/lego_dataset")

    index = pd.read_csv( 'C:/Users/dchro/Documents/MLOps/mlops_project/data/external/lego_dataset/index.csv')
    labels = index["class_id"]-1
    files = index["path"]

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    trainset = Lego_Dataset(file_paths=files, path = path, labels=labels,transform=transform)

    torch.save(trainset, 'data/processed/LEGO_torch_train_dataset.pt')