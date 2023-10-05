import torch
from torch.utils.data import DataLoader
from src.data.make_dataset import make_dataset,get_transform,convert_label
from src import _PROJECT_ROOT
import os
from timm import create_model
import hydra
import numpy as np
from omegaconf import DictConfig
from PIL import Image



def load_model(model_path):
    model = create_model("mobilenetv3_large_100", pretrained=False, num_classes=38)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode
    return model


def predict_test(config):
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

def predict(config : DictConfig,img : Image) -> str:

    model = load_model(
        model_path=os.path.join(_PROJECT_ROOT, "models", "mobilenetv3_fine_tuned.pth")
    )

    config

    with torch.no_grad():
        transforms = get_transform(config, dataset_type= "test")
        image_rgb = img.convert("RGB")
        processed = transforms(image_rgb).unsqueeze(0)
        outputs = model(processed)
        class_id = torch.argmax(outputs, dim=1).cpu().detach().numpy()[0]
    prediction = convert_label(class_id)
    return prediction



@hydra.main(config_path="../../config/", config_name="main.yaml")
def main(cfg):
    prediction = predict(cfg,Image.open('C:/Users/Lennart/Documents/GitHub/mlops_project/data/external/lego_dataset/test/005.jpg'))
    print(prediction)



main()