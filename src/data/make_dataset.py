# -*- coding: utf-8 -*-
import pandas as pd
import torch
import os
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import logging
from torchvision import transforms

import torch
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

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]

        return image, label




def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    #ToDo Paths

    parent = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    data_path = os.path.join(parent,'data/external/lego_dataset/')


    index = pd.read_csv(os.path.join(data_path,Path('index.csv')))
    labels = index["class_id"]-1
    files = index["path"]

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])



    trainset = Lego_Dataset(file_paths=files, path = data_path, labels=labels,transform=transform)

    torch.save(trainset,os.path.join(parent,Path('data/processed/trainset.pth')))
    #train_loader = DataLoader(trainset, batch_size=32, shuffle=True)

    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
