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
root_directory = os.path.abspath(os.path.join(parent_directory, '..'))
sys.path.append(parent_directory)

from data.make_dataset import Lego_Dataset


def get_data():
    data_path = os.path.join(root_directory,"data", "external", "lego_dataset")

    index = pd.read_csv(os.path.join(data_path, 'index.csv'))
    labels = index["class_id"]-1
    files = index["path"]

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    return Lego_Dataset(file_paths=files, path = data_path, labels=labels,transform=transform)

def train_model(num_epochs = 2, lr = 0.003, criterion = nn.CrossEntropyLoss()):
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
                "EPOCH: {:5}    BATCH: {:5}/{:5}    LOSS: {:.3f}".format(
                    ep, batch_idx, len(train_loader), batch_loss
                )
            )

        epoch_loss = total_loss / len(trainset)
        epoch_accuracy = num_correct / len(trainset)
        print(
            "EPOCH: {:5}    LOSS: {:.3f}    ACCURACY: {:.3f}".format(
                ep, epoch_loss, epoch_accuracy
            )
        )

    # Save the trained model
    torch.save(model.state_dict(), os.path.join(root_directory, 'models', 'mobilenetv3_fine_tuned.pth'))
    print("Model saved!")


# Run training, save model and print metrics
train_model()