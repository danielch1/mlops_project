import os
from typing import List, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from scipy import stats

from src import _PROJECT_ROOT

# Define the paths to your training and test data directories


# Function to calculate average brightness and contrast for a set of images using Pillow
def calculate_stats(
    data_dir: str, file_paths: List[str]
) -> Tuple[List[float], List[float]]:
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

    return avg_brightness_values, avg_contrast_values


# Calculate statistics for the training and test sets separately

data_path = os.path.join(_PROJECT_ROOT, "data", "external", "lego_dataset")
test_data_path = os.path.join(_PROJECT_ROOT, "data", "external", "lego_dataset", "test")

train_index_file = pd.read_csv(os.path.join(data_path, "index.csv"))
test_index_file = pd.read_csv(os.path.join(data_path, "test.csv"))


train_avg_brightness, train_contrast = calculate_stats(
    data_path, train_index_file["path"]
)
test_avg_brightness, test_contrast = calculate_stats(data_path, test_index_file["path"])

t_statistic_brightness, p_value_brightness = stats.ttest_ind(
    train_avg_brightness, test_avg_brightness
)
t_statistic_contrast, p_value_contrast = stats.ttest_ind(train_contrast, test_contrast)

alpha = 0.05

print(p_value_brightness > alpha)
print(p_value_contrast > alpha)
# Compare the statistics
