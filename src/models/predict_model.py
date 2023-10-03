import torch
from torch.utils.data import DataLoader
from src.data.make_dataset import Lego_Dataset
import os
from timm import create_model
import hydra
import numpy as np



def load_model(model_path):
    model = create_model('mobilenetv3_large_100', pretrained= False, num_classes=38)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode
    return model

#@hydra.main(config_name="model_config.yaml")
def predict(model, test_dataset):
    dataloader = DataLoader(test_dataset, batch_size = 32, shuffle=False)
    predictions = []

    with torch.no_grad():
        for batch_idx, (batch,_) in enumerate(dataloader):
            inputs = batch
            outputs = model(inputs)
            predictions.extend(torch.argmax(outputs, dim=1).cpu().detach().numpy())
        
    return predictions


if __name__ == "__main__":

    current_script_directory = os.path.dirname(os.path.abspath(__file__))
    parent_directory = os.path.abspath(os.path.join(current_script_directory, '..'))
    root_directory = os.path.abspath(os.path.join(parent_directory, '..'))

    model_path = os.path.join(root_directory,"models", "mobilenetv3_fine_tuned.pth")


    model = load_model(model_path)
    test_data = torch.load(os.path.join(root_directory,"data", "processed", "test_dataset.pth"))
    test_data.set_path()
    predictions = predict(model, test_data)
    #labels = np.array([get_test_Data()[i][1] for i in range(len(get_test_Data()))])

    #ToDo add Image name possibly image visualization
    np.savetxt(os.path.join(root_directory,"src","models","predictions.csv"), predictions, delimiter=',')
