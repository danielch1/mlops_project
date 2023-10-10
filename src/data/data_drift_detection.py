import os
from typing import List, Tuple

import numpy as np
import pandas as pd
from PIL import Image

from src import _PROJECT_ROOT

# Define the paths to your training and test data directories


# Function to calculate average brightness and contrast for a set of images using Pillow
def calculate_stats(data_dir: str, file_paths: List[str]) -> Tuple[float, float]:
    # Initialize lists to store the average brightness and contrast values
    avg_brightness_values = []
    avg_contrast_values = []

    # Iterate over each image in the directory
    for filename in file_paths:
        if filename.endswith(".jpg"):
            # Load the image using Pillow
            image_path = os.path.join(data_dir, filename)
            image = Image.open(image_path)

            # Convert the image to grayscale
            gray_image = image.convert("L")

            # Calculate average brightness and contrast
            pixel_array = np.array(gray_image)
            avg_brightness = np.mean(pixel_array)
            avg_contrast = np.std(pixel_array)

            # Append to the lists
            avg_brightness_values.append(avg_brightness)
            avg_contrast_values.append(avg_contrast)

    # Calculate mean average brightness and mean contrast
    mean_avg_brightness = np.mean(avg_brightness_values)
    mean_avg_contrast = np.mean(avg_contrast_values)

    return mean_avg_brightness, mean_avg_contrast


# Calculate statistics for the training and test sets separately

data_path = os.path.join(_PROJECT_ROOT, "data", "external", "lego_dataset")
test_data_path = os.path.join(_PROJECT_ROOT, "data", "external", "lego_dataset", "test")

train_index_file = pd.read_csv(os.path.join(data_path, "index.csv"))
test_index_file = pd.read_csv(os.path.join(data_path, "test.csv"))


train_mean_brightness, train_mean_contrast = calculate_stats(
    data_path, train_index_file["path"]
)
test_mean_brightness, test_mean_contrast = calculate_stats(
    data_path, test_index_file["path"]
)

# Compare the statistics
print(
    f"Training Set - Avg Brightness: {train_mean_brightness}, Avg Contrast: {train_mean_contrast}"
)
print(
    f"Test Set - Avg Brightness: {test_mean_brightness}, Avg Contrast: {test_mean_contrast}"
)
