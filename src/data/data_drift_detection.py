import datetime
import io
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import requests
from google.cloud import storage
from PIL import Image
from scipy import stats

client = storage.Client()
bucket = client.get_bucket("mlops-project-ss2023-bucket")


def load_cloud_index(inf_idx: bool = True) -> pd.DataFrame:
    index_file_name = "lego_dataset/index.csv"
    if inf_idx:
        index_file_name = "lego_dataset/inference_index.csv"

    index_blob = bucket.blob(index_file_name)
    csv_content = index_blob.download_as_text()

    # Read the CSV content into a DataFrame
    df = pd.read_csv(io.StringIO(csv_content))

    return df


def save_cloud_drift_log(df: pd.DataFrame, date_time: str) -> None:
    destination_file_name = f"data_drifting_logs/{date_time}_log.csv"
    csv_data = io.StringIO()
    df.to_csv(csv_data, index=False)

    blob = bucket.blob(destination_file_name)
    blob.upload_from_string(csv_data.getvalue())


# Function to calculate average brightness and contrast for a set of images using Pillow
def calculate_stats(file_paths: List[str]) -> Tuple[List[float], List[float]]:
    # Initialize lists to store the average brightness and contrast values
    avg_brightness_values = []
    avg_contrast_values = []
    # Iterate over each image in the directory
    for file_path in file_paths:
        if file_path.endswith(".jpg"):
            blob = bucket.blob(os.path.join("lego_dataset", file_path))

            # Download the file as bytes
            image_data = blob.download_as_bytes()

            # Convert the downloaded bytes into a PIL.Image object
            image = Image.open(io.BytesIO(image_data))

            # Convert the image to grayscale
            gray_image = image.convert("L")

            # Calculate average brightness and contrast
            pixel_array = np.array(gray_image)
            avg_brightness = np.mean(pixel_array)
            avg_contrast = np.std(pixel_array)

            # Append to the lists
            avg_brightness_values.append(avg_brightness)
            avg_contrast_values.append(avg_contrast)

    return avg_brightness_values, avg_contrast_values


# Calculate statistics for the training and test sets separately
def data_drift_detection():
    train_idx = load_cloud_index(inf_idx=False)["path"]
    train_avg_brightness, train_contrast = calculate_stats(train_idx)

    inf_idx = load_cloud_index(inf_idx=True)["path"]
    inf_avg_brightness, inf_contrast = calculate_stats(inf_idx)

    train_mean_brightness = np.mean(train_avg_brightness)
    train_mean_contrast = np.mean(train_contrast)

    inf_mean_brightness = np.mean(inf_avg_brightness)
    inf_mean_contrast = np.mean(inf_contrast)

    t_statistic_brightness, p_value_brightness = stats.ttest_ind(
        train_avg_brightness, inf_avg_brightness
    )
    t_statistic_contrast, p_value_contrast = stats.ttest_ind(
        train_contrast, inf_contrast
    )

    df = pd.DataFrame(
        data={
            "train_mean_brightness": train_mean_brightness,
            "train_mean_contrast": train_mean_contrast,
            "inf_mean_brightness": inf_mean_brightness,
            "inf_mean_contrast": inf_mean_contrast,
            "p_value_brightness": p_value_brightness,
            "p_value_contrast": p_value_contrast,
        },
        index=[0],
    )
    current_datetime = datetime.datetime.now()
    formatted_datetime = str(current_datetime.strftime("%Y-%m-%d_%H:%M:%S"))
    save_cloud_drift_log(df, formatted_datetime)

    return p_value_brightness, p_value_contrast


# data_drift_detection()


# Replace with the URL of your Google Cloud Function
function_url = (
    "https://europe-west3-mlops-project-401218.cloudfunctions.net/data_drift_detection"
)

# Make an HTTP GET request to trigger the function
response = requests.get(function_url)

# Check if the request was successful (status code 200)
if response.status_code == 200:
    data = response.json()
    p_value_brightness = data["p_value_brightness"]
    p_value_contrast = data["p_value_contrast"]
    print("p_value_brightness:", p_value_brightness)
    print("p_value_contrast:", p_value_contrast)
else:
    print("HTTP GET request failed with status code:", response.status_code)
