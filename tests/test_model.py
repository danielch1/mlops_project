from tests import _PROJECT_ROOT
import torch
import os
from src.models.predict_model import load_model


def test_output_shape():
    model = load_model(os.path.join(_PROJECT_ROOT,'models','mobilenetv3_fine_tuned.pth'))
    model.eval()
    output = model(torch.zeros(1,3,224,224))
    assert output.shape == torch.Size([1,38]) , 'Model doesent generate the expected output shape'
