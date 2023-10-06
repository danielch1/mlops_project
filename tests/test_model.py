import os

import pytest
import torch

from src.models.predict_model import load_model
from tests import _PROJECT_ROOT

# Define the path to the file you want to check for existence
file_to_check = os.path.join(_PROJECT_ROOT, "models", "mobilenetv3_fine_tuned.pth")


# Use the pytest.mark.skipif decorator to conditionally skip the test
@pytest.mark.skipif(not os.path.exists(file_to_check), reason="File not found")
def test_output_shape():
    model = load_model(file_to_check)
    model.eval()
    output = model(torch.zeros(1, 3, 224, 224))
    assert output.shape == torch.Size(
        [1, 38]
    ), "Model doesn't generate the expected output shape"
