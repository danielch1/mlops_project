import io
import os
import re
import tempfile
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from google.cloud import storage


def load_cloud_index() -> pd.DataFrame:
    index_file_name = "lego_dataset/test_batch.csv"

    index_blob = bucket.blob(index_file_name)
    csv_content = index_blob.download_as_text()

    # Read the CSV content into a DataFrame
    df = pd.read_csv(io.StringIO(csv_content))

    return df


def get_list_of_images_in_bucket(i):
    list_of_images = []
    prefix_to_match = f"lego_dataset/test_batch{i}/"
    for blob in blobs:
        if blob.name.startswith(prefix_to_match) and not blob.name.endswith("/"):
            list_of_images.append(blob.name)
    return list_of_images


def get_prediction(api_url, bucket, image_blob_name):
    max_retries = 3  # Maximum number of retries
    retries = 0

    while retries < max_retries:
        # Get the blob (image) from the bucket
        blob = bucket.blob(image_blob_name)

        # Download the image to the temporary directory
        temp_dir = tempfile.mkdtemp()
        temp_image_path = os.path.join(temp_dir, "temp_image.jpg")
        blob.download_to_filename(temp_image_path)

        # Prepare the image file for the request
        with open(temp_image_path, "rb") as file:
            files = {"image": (image_blob_name, file, "image/jpeg")}
            response = requests.post(api_url, files=files)

        # Clean up the temporary file if needed
        os.remove(temp_image_path)
        os.rmdir(temp_dir)

        # Process the response
        if response.status_code == 200:
            prediction = response.json()["prediction"]
            retries = 0
            return prediction
        elif response.status_code == 500:
            print("Received a 500 error. Retrying...")
            retries += 1
            time.sleep(1)  # Add a delay before retrying
        else:
            print(f"Error: {response.status_code}")
            break

    print("Max retries reached. Unable to get a successful response.")
    return None


def get_true_class(image_blob_name, df_true_classes):
    # Remove first directory from the path
    path_parts = image_blob_name.split("/")
    new_path = "/".join(path_parts[1:])

    return df_true_classes[df_true_classes["path"] == new_path]["class_id"].values[0]


def compare_prediction_to_true_class(prediction, true_class):
    # Extract the number from the prediction string
    match = re.search(r"\d+", prediction)
    number_prediction = float(match.group())

    return number_prediction == true_class


def save_results(results_all, bucket):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    df = pd.DataFrame(results_all)
    csv_path = f"results_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    blob_csv = bucket.blob(f"performance_logs/results_{timestamp}.csv")
    blob_csv.upload_from_filename(csv_path)

    # Calculate the averages for each inner list
    averages = [np.mean(inner_list) for inner_list in results_all]

    # Create a bar plot
    plt.bar(
        range(len(averages)), averages, tick_label=[f"Batch {j + 1}" for j in range(i)]
    )
    plt.xlabel("Lists")
    plt.ylabel("Prediction Accuracy")
    plt.title("Accuracy for batch of data")

    # Save the plot with a timestamp in the filename locally
    plot_filename = f"accuracy_plot_{timestamp}.png"
    plt.savefig(plot_filename)

    # Send file to a bucket
    blob_plot = bucket.blob(f"performance_logs/accuracy_plot_{timestamp}.png")
    blob_plot.upload_from_filename(plot_filename)

    # Clean temp files
    os.remove(plot_filename)
    os.remove(csv_path)

    return None


client = storage.Client()
bucket = client.get_bucket("mlops-project-ss2023-bucket")
blobs = list(bucket.list_blobs())

df_true_classes = load_cloud_index()

API_URL = "https://inference-mmhol5imca-ey.a.run.app/predict/"


# Iterrating over all data batches in a bucket
i = 0
results_all = []
while True:

    results_batch = []
    list_of_images = get_list_of_images_in_bucket(i)

    if len(list_of_images) == 0:
        break

    for image_blob_name in list_of_images:
        print(f"image_blob_name: {image_blob_name}")
        prediction = get_prediction(API_URL, bucket, image_blob_name)
        # print(f"prediction: {prediction}")

        # print(f'image_blob_name: {image_blob_name}')
        true_class = get_true_class(image_blob_name, df_true_classes) - 1
        # print(f"True class: {true_class}")

        is_true = compare_prediction_to_true_class(prediction, true_class)
        # print(is_true)
        results_batch.append(int(is_true))

    i += 1
    results_all.append(results_batch)


print("Assessment Done! \n \t Saving results...")

# Save results in an plot to csv file

save_results(results_all, bucket)
