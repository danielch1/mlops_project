import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import timm
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Import LegoDataset class (automated relative import)
current_script_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.abspath(os.path.join(current_script_directory, '..'))
sys.path.append(parent_directory)

from data.make_dataset import Lego_Dataset


def get_data():
    # Load data
    folder_path = 'data/processed/'  # Replace with the actual folder path
    file_name = 'LEGO_torch_train_dataset.pickle'  # Replace with the actual file name
    full_path = os.path.join(folder_path, file_name)

    wd = os.getcwd()
    path = os.path.join(wd,"C:/Users/dchro/Documents/MLOps/mlops_project/data/external/lego_dataset")

    index = pd.read_csv('C:/Users/dchro/Documents/MLOps/mlops_project/data/external/lego_dataset/index.csv')
    labels = index["class_id"]-1
    files = index["path"]

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    return Lego_Dataset(file_paths=files, path = path, labels=labels,transform=transform)

def train_model(num_epochs = 3, lr = 0.003, criterion = nn.CrossEntropyLoss()):
    # Data Load
    print("Loading data...")
    num_classes = 38
    trainset = get_data()
    train_loader = DataLoader(trainset, batch_size=32, shuffle=True) 

    print("Defining model...")
    # Model definition
    model = timm.create_model('mobilenetv3_large_100', pretrained=True, num_classes=num_classes)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print("Training start...")
    # Training loop
    for ep in range(num_epochs):
        total_loss = 0
        num_correct = 0

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            model.train()
            optimizer.zero_grad()
            y_hat = model(inputs)
            batch_loss = criterion(y_hat, labels)
            batch_loss.backward()
            optimizer.step()

            total_loss += float(batch_loss)
            num_correct += int(torch.sum(torch.argmax(y_hat, dim=1) == labels))

            
            print(
                "EPOCH: {:5}/tBATCH: {:5}/{:5}/tLOSS: {:.3f}".format(
                    ep, batch_idx, len(train_loader), batch_loss
                )
            )

        epoch_loss = total_loss / len(trainset)
        epoch_accuracy = num_correct / len(trainset)
        print(
            "EPOCH: {:5} /t LOSS: {:.3f} /t ACCURACY: {:.3f}".format(
                ep, epoch_loss, epoch_accuracy
            )
        )


        
        # Validation loop (optional)
        #model.eval()
        #with torch.no_grad():
        #    for inputs, labels in val_loader:
        #        outputs = model(inputs)
                # Calculate validation loss and metrics

    # Save the trained model
    torch.save(model.state_dict(), 'models/mobilenetv3_fine_tuned.pth')
    print("Model saved!")


# Run training, save model and print metrics
train_model()


