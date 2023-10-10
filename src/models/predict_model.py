import io
import os

import pandas as pd
import torch
from fastapi import FastAPI, UploadFile
from google.cloud import storage
from hydra import compose, initialize
from PIL import Image
from pydantic import BaseModel
from timm import create_model

from src.data.make_dataset import convert_label, get_transform

# def predict_test(config):
#     dataloader = DataLoader(
#         make_dataset(config, dataset_type="test"), batch_size=32, shuffle=False
#     )
#     predictions = []

#     model = load_model(
#         model_path=os.path.join(_PROJECT_ROOT, "models", "mobilenetv3_fine_tuned.pth")
#     )

#     with torch.no_grad():
#         for batch_idx, (batch, _) in enumerate(dataloader):
#             inputs = batch
#             outputs = model(inputs)
#             predictions.extend(torch.argmax(outputs, dim=1).cpu().detach().numpy())

#     return predictions


client = storage.Client()
bucket = client.get_bucket("mlops-project-ss2023-bucket")


def load_cloud_index() -> pd.DataFrame:
    index_file_name = "lego_dataset/inference_index.csv"

    index_blob = bucket.blob(index_file_name)
    csv_content = index_blob.download_as_text()

    # Read the CSV content into a DataFrame
    df = pd.read_csv(io.StringIO(csv_content))

    return df


def save_cloud_index(df: pd.DataFrame):
    destination_file_name = "lego_dataset/inference_index.csv"
    csv_data = io.StringIO()
    df.to_csv(csv_data, index=False)

    blob = bucket.blob(destination_file_name)
    blob.upload_from_string(csv_data.getvalue())


def save_cloud_image(image: Image, idx: str) -> None:
    name = f"{idx}.jpg"
    destination_image_name = os.path.join("lego_dataset", "inference_images", name)

    image_stream = io.BytesIO()
    image.save(image_stream, format="JPEG")  # You can specify the format you need

    # Upload the image from the BytesIO stream to the GCS bucket
    blob = bucket.blob(destination_image_name)
    image_stream.seek(0)
    blob.upload_from_file(image_stream, content_type="image/jpeg")


def capture_inference_image(image: Image, pred_label: int):
    df = load_cloud_index()
    idx = str(len(df.index) + 1)
    new_line = pd.DataFrame(
        data={
            "path": [os.path.join("inference_images", f"{idx}.jpg")],
            "class_id": [pred_label],
        }
    )
    df = pd.concat((df, new_line))
    save_cloud_index(df)
    save_cloud_image(image, idx)


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
            capture_inference_image(img, class_id)

        prediction = convert_label(class_id)
    return {"prediction": prediction}
