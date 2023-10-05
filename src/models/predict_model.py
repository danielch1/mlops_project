import torch
from torch.utils.data import DataLoader
from src.data.make_dataset import make_dataset,get_transform,convert_label
from src import _PROJECT_ROOT
import os
from timm import create_model
import hydra
import numpy as np
from omegaconf import DictConfig
from fastapi import FastAPI, UploadFile
from pydantic import BaseModel
from PIL import Image
import io
from hydra import initialize, compose


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

class PredictionRequest(BaseModel):
    image: UploadFile


class PredictionResponse(BaseModel):
    prediction: str

app = FastAPI()

@app.post("/predict/", response_model=PredictionResponse)
async def predict_image(image: UploadFile):
    with initialize(version_base=None, config_path='../config/'):
        cfg = compose(config_name= 'main.yaml')
        model = load_model(
            model_path=os.path.join(_PROJECT_ROOT, "models", "mobilenetv3_fine_tuned.pth")
        )

        with torch.no_grad():
            transforms = get_transform(config, dataset_type="test")
            img_bytes = await image.read()
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            processed = transforms(img)
            outputs = model(processed.unsqueeze(0))
            class_id = torch.argmax(outputs, dim=1).cpu().detach().numpy()
        
        prediction = convert_label(class_id)
    return {"prediction": prediction}



#@hydra.main(config_path="../../config/", config_name="main.yaml")
#def main(cfg):
#    prediction = predict(cfg,Image.open('C:/Users/Lennart/Documents/GitHub/mlops_project/data/external/lego_dataset/test/005.jpg'))
#    print(prediction)



#main()