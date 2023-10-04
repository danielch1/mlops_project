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
from src.data.make_dataset import Lego_Dataset


current_script_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.abspath(os.path.join(current_script_directory, '..'))
root_directory = os.path.abspath(os.path.join(parent_directory, '..'))

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


@hydra.main(config_path='../../config/experiment', config_name="main.yaml")
def train_model(cfg):
    # Data Load
    print("Loading data...")
    num_classes = 38
    trainset = torch.load(os.path.join(root_directory,"data", "processed", "train_dataset.pth"))
    trainset.set_path()
    val_set = torch.load(os.path.join(root_directory,"data", "processed", "val_dataset.pth"))
    val_set.set_path()
    train_loader = DataLoader(trainset, batch_size=cfg.hparams.batch_size, shuffle=cfg.hparams.shuffle)
    val_loader = DataLoader(val_set, batch_size= cfg.hparams.batch_size, shuffle = False)

    print("Defining model...")
    # Model definition
    model = timm.create_model('mobilenetv3_large_100', pretrained=cfg.hparams.take_pretrained_model, num_classes=num_classes)
    optimizer = optim.Adam(model.parameters(), lr=cfg.hparams.learning_rate)
    criterion = nn.CrossEntropyLoss()


    best_val_loss = float('inf')
    patience = 5  # Number of epochs to wait for improvement
    epochs_without_improvement = 0

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

        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            epochs_without_improvement = 0
            
            #save the best model checkpoint here
            torch.save(model.state_dict(), os.path.join(root_directory, 'models', 'mobilenetv3_fine_tuned.pth'))
        else:
            epochs_without_improvement += 1

        # Check if we should stop training
        if epochs_without_improvement >= patience:
            print(f"Early stopping after {ep + 1} epochs")
            break

    # Save the trained model
    
    print("Best Model saved!")


# Run training, save model and print metrics
train_model()