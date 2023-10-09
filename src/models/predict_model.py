import io
import os

import torch
from fastapi import FastAPI, UploadFile
from google.cloud import storage
from hydra import compose, initialize
from PIL import Image
from pydantic import BaseModel
from timm import create_model
from torch.utils.data import DataLoader

from src import _PROJECT_ROOT
from src.data.make_dataset import convert_label, get_transform, make_dataset


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
    storage_client = storage.Client()
    bucket_name = "mlops-project-ss2023-bucket"

    model_blob_name = "mobilenetv3_fine_tuned.pth"

    with initialize(version_base=None, config_path="../../config/"):
        cfg = compose(config_name="main.yaml")
        model_blob_name = "mobilenetv3_fine_tuned.pth"

        # Load the model state dictionary from Google Cloud Storage
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(model_blob_name)
        model_bytes = blob.download_as_bytes()

        # Load  PyTorch model with state dictionary
        model = create_model("mobilenetv3_large_100", pretrained=False, num_classes=38)
        model.load_state_dict(
            torch.load(io.BytesIO(model_bytes), map_location=torch.device("cpu"))
        )
        model.eval()

        with torch.no_grad():
            transforms = get_transform(cfg, dataset_type="test")
            img_bytes = await image.read()
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            processed = transforms(img)
            outputs = model(processed.unsqueeze(0))
            class_id = torch.argmax(outputs, dim=1).cpu().detach().numpy()[0]

        prediction = convert_label(class_id)
    return {"prediction": prediction}
