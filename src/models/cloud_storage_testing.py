import io
import os

import pandas as pd
from google.cloud import storage
from PIL import Image

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


image_path = (
    "/Users/LeMarx/Documents/01_Projects/mlops_project/"
    "data/external/lego_dataset/test/058.jpg"
)

img = Image.open(image_path)

capture_inference_image(img, 10)

print(load_cloud_index())
