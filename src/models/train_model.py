import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import timm
import hydra
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import wandb

wandb.init(
    # set the wandb project where this run will be logged
    project="my-awesome-project",
    
    # track hyperparameters and run metadata
    config={
    "learning_rate": 0.003,
    "architecture": "mobilenetv3",
    "dataset": "Lego-Minifigures",
    "epochs": 10,
    }
)

# Import LegoDataset class (automated relative import)
current_script_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.abspath(os.path.join(current_script_directory, '..'))
root_directory = os.path.abspath(os.path.join(parent_directory, '..'))
sys.path.append(parent_directory)

from data.make_dataset import Lego_Dataset


def get_data():
    data_path = os.path.join(root_directory,"data", "external", "lego_dataset")

    index = pd.read_csv(os.path.join(data_path, 'index.csv'))


    #Train Validation Split
    train_index = index.sample(int(0.75*len(index.index)))

    remaining_indices = list(set(index.index) - set(train_index.index))
    # Create a new DataFrame with the remaining indices
    train_index.reset_index(inplace= True, drop=True)
    val_index = index.loc[remaining_indices].reset_index(drop=True)

    train_labels = train_index["class_id"]-1
    train_files = train_index["path"]

    val_labels = val_index["class_id"]-1
    val_files = val_index["path"]


    train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),  # Random crop and resize to 224x224
    transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
    transforms.RandomRotation(degrees=30),  # Randomly rotate the image up to 30 degrees
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Randomly translate the image
    transforms.RandomPerspective(distortion_scale=0.5, p=0.5),  # Apply perspective transformation
    transforms.ToTensor(),  # Convert the image to a PyTorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet statistics
])

    # Define transforms for validation and test data (typically no augmentation)
    val_transforms = transforms.Compose([
        transforms.Resize(256),  # Resize to 256x256
        transforms.CenterCrop(224),  # Center crop to 224x224
        transforms.ToTensor(),  # Convert the image to a PyTorch tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet statistics
    ])


    return Lego_Dataset(file_paths=train_files, path = data_path, labels=train_labels,transform=train_transforms),Lego_Dataset(file_paths=val_files, path = data_path, labels=val_labels,transform=val_transforms)

@hydra.main(config_name="model_config.yaml")
def train_model(cfg):
    # Data Load
    print("Loading data...")
    num_classes = 38
    trainset,val_set = get_data()
    train_loader = DataLoader(trainset, batch_size=cfg.hparams.batch_size, shuffle=cfg.hparams.shuffle)
    val_loader = DataLoader(trainset, batch_size= cfg.hparams.batch_size, shuffle = cfg.hparams.shuffle)

    print("Defining model...")
    # Model definition
    model = timm.create_model('mobilenetv3_large_100', pretrained=cfg.hparams.take_pretrained_model, num_classes=num_classes)
    optimizer = optim.Adam(model.parameters(), lr=cfg.hparams.learning_rate)
    criterion = nn.CrossEntropyLoss()

    print("Training start...")
    # Training loop
    for ep in range(cfg.hparams.num_epochs):
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
            
        model.eval()
        total_val_loss = 0
        num_val_correct = 0
        
        with torch.no_grad():
            for batch_idx, (val_inputs, val_labels) in enumerate(val_loader):
                val_outputs = model(val_inputs)
                val_loss = criterion(val_outputs, val_labels)
                total_val_loss += float(val_loss)
                num_val_correct += int(torch.sum(torch.argmax(val_outputs, dim=1) == val_labels))


        val_epoch_loss = total_val_loss / len(val_set)
        val_epoch_accuracy = num_val_correct / len(val_set)
        print(
            "EPOCH: {:5}    VAL LOSS: {:.3f}    VAL ACCURACY: {:.3f}".format(
                ep, val_epoch_loss, val_epoch_accuracy
                )
        )

        wandb.log({'train_acc' : epoch_accuracy,'val_acc' : val_epoch_accuracy,'train_loss' : batch_loss,'val_loss' : val_loss})

    # Save the trained model
    torch.save(model.state_dict(), os.path.join(root_directory, 'models', 'mobilenetv3_fine_tuned.pth'))
    print("Model saved!")


# Run training, save model and print metrics
train_model()