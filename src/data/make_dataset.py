# -*- coding: utf-8 -*-
import pandas as pd
import torch
import os
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from src.data import Lego_Dataset
import logging
from torchvision import transforms


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


    torch.save(trainset, os.path.join(parent,'/data/processed/trainset.pth'))
    #train_loader = DataLoader(trainset, batch_size=32, shuffle=True)

    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    #load_dotenv(find_dotenv())

    main()
