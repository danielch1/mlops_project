import os
import time

import hydra
import timm
import torch
import torch.nn as nn
import torch.optim as optim
from dotenv import load_dotenv
from torch.utils.data import DataLoader

from src.data.make_dataset import make_dataset

current_script_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.abspath(os.path.join(current_script_directory, ".."))
root_directory = os.path.abspath(os.path.join(parent_directory, ".."))
save_path = os.path.join(root_directory, "models", "mobilenetv3_fine_tuned.pth")

# Initialize wandb with your API key and project name (including the team name)
# Load environment variables from .env file
load_dotenv()


# @hydra.main(config_path="../../config/", config_name="main.yaml")
def train_model(cfg):
    # Data Load
    print("Loading data...")
    num_classes = 38
    trainset = make_dataset(cfg, dataset_type="train")
    val_set = make_dataset(cfg, dataset_type="val")
    train_loader = DataLoader(
        trainset,
        batch_size=cfg.experiment.hparams.batch_size,
        shuffle=cfg.experiment.hparams.shuffle,
        num_workers=i,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg.experiment.hparams.batch_size,
        shuffle=False,
        num_workers=i,
    )

    print("Defining model...")
    # Model definition
    model = timm.create_model(
        "mobilenetv3_large_100", pretrained=True, num_classes=num_classes
    )
    optimizer = optim.Adam(model.parameters(), lr=cfg.experiment.hparams.learning_rate)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    epochs_without_improvement = 0

    print("Training start...")
    # Training loop
    for ep in range(cfg.experiment.hparams.num_epochs):
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
                num_val_correct += int(
                    torch.sum(torch.argmax(val_outputs, dim=1) == val_labels)
                )

        val_epoch_loss = total_val_loss / len(val_set)
        val_epoch_accuracy = num_val_correct / len(val_set)
        print(
            "EPOCH: {:5}    VAL LOSS: {:.3f}    VAL ACCURACY: {:.3f}".format(
                ep, val_epoch_loss, val_epoch_accuracy
            )
        )

        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            epochs_without_improvement = 0

            # save the best model checkpoint here
            torch.save(model.state_dict(), save_path)
        else:
            epochs_without_improvement += 1

        # # Check if we should stop training
        # if epochs_without_improvement >= patience:
        #     print(f"Early stopping after {ep + 1} epochs")
        #     break

    # Save the trained model

    print("Best Model saved!")
    print("Save path: ", save_path)


# Run training, save model and print metrics
@hydra.main(config_path="../../config/", config_name="main.yaml")
def main(cfg):
    train_model(cfg)


if __name__ == "__main__":

    results_list = []

    for i in [1, 2, 4]:
        # Print time for each run
        print(f"Running with {i} workers")
        start = time.time()

        main()

        end = time.time()
        print(f"Time taken: {round(end - start, 2)}")
        results_list.append([f"Num_workers: {i}", f"Time: {round(end - start, 2)}"])

    # Print results
    print(f"Results: \n {results_list}")
