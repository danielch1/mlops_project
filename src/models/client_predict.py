import requests

# Replace with the path to your image file
image_path = (
    "/Users/LeMarx/Documents/01_Projects/mlops_project/"
    "data/external/lego_dataset/test/005.jpg"
)
url = "http://127.0.0.1:8000/predict/"

# Send a POST request with the image file
with open(image_path, "rb") as file:
    files = {"image": (image_path, file, "image/jpeg")}
    response = requests.post(url, files=files)

# Parse the JSON response to obtain the prediction
if response.status_code == 200:
    prediction = response.json()["prediction"]
    print(f"Prediction: {prediction}")
else:
    print(f"Error: {response.status_code}")
