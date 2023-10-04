import torch
from torch.utils.data import DataLoader
from src.data.make_dataset import make_dataset
from src import _PROJECT_ROOT
import os
from timm import create_model
import hydra
import numpy as np


def load_model(model_path):
    model = create_model("mobilenetv3_large_100", pretrained=False, num_classes=38)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode
    return model


def predict(config):
    dataloader = DataLoader(
        make_dataset(config, dataset_type="test"), batch_size=32, shuffle=False
    )
    predictions = []

    model = load_model(
        model_path=os.path.join(_PROJECT_ROOT, "models", "mobilenetv3_fine_tuned.pth")
    )

    with torch.no_grad():
        for batch_idx, (batch, _) in enumerate(dataloader):
            inputs = batch
            outputs = model(inputs)
            predictions.extend(torch.argmax(outputs, dim=1).cpu().detach().numpy())

    return predictions


@hydra.main(config_path="../../config/", config_name="main.yaml")
def main(cfg):
    predictions = predict(cfg)
    save_path = os.path.join(_PROJECT_ROOT, "src", "models", "predictions.csv")
    np.savetxt(save_path, predictions, delimiter=",")
