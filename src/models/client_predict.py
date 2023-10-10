import os
import random

import matplotlib.pyplot as plt
import requests
from PIL import Image

from src import _PROJECT_ROOT

# Random choice of an image to classification
test_path = os.path.join(_PROJECT_ROOT, "data", "external", "lego_dataset", "test")
all_items = os.listdir(test_path)
random_item = random.choice(all_items)
image_path = os.path.join(test_path, random_item)

# URL for the API
url = "https://inference-mmhol5imca-ey.a.run.app/predict/"
# url = "http://127.0.0.1:8000/predict"


# Draw a test image
def draw_test_image(all_items):
    random_item = random.choice(all_items)
    return os.path.join(test_path, random_item)


# Send a POST request with the image file
def send_predict_request(url, image_path):
    with open(image_path, "rb") as file:
        files = {"image": (image_path, file, "image/jpeg")}
        response = requests.post(url, files=files)

    return response


# Process the response, dislpay classified image
def process_response(response):
    if response.status_code == 200:
        prediction = response.json()["prediction"]
        print(f"Prediction: {prediction}")

        # Display the image and the prediction
        image = Image.open(image_path)
        plt.imshow(image)
        plt.title(f"Image classified as: {prediction}")
        plt.axis("off")
        plt.show()
    else:
        print(f"Error: {response.status_code}")

    return None


if __name__ == "__main__":
    image_path = draw_test_image(all_items)
    response = send_predict_request(url, image_path)
    process_response(response)
