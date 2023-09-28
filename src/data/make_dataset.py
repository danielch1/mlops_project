# -*- coding: utf-8 -*-
import pandas as pd
import torch
import os
from PIL import Image
from torchvision import transforms
from LegoDataset import Lego_Dataset

if __name__ == '__main__':

    wd = os.getcwd()
    path = os.path.join(wd,"C:/Users/dchro/Documents/MLOps/mlops_project/data/external/lego_dataset")

    index = pd.read_csv( 'C:/Users/dchro/Documents/MLOps/mlops_project/data/external/lego_dataset/index.csv')
    labels = index["class_id"]-1
    files = index["path"]

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    trainset = Lego_Dataset(file_paths=files, path = path, labels=labels,transform=transform)

    trainset.save('data/processed/LEGO_torch_train_dataset.pt')
    trainset.load('data/processed/LEGO_torch_train_dataset.pt')
