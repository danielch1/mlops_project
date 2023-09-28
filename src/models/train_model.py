import torch
import torch.nn as nn
import torch.optim as optim
import timm
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

num_epochs = 10
lr = 0.001

train_loader = DataLoader()
val_loader = DataLoader() #ToDo
num_classes = 10 #ToDO


model = timm.create_model('mobilenetv3_small_100', pretrained=True, num_classes=num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

print(model.pretrained_cfg)

# Adjust this according to your needs
for epoch in range(num_epochs):
    model.train()
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
    # Validation loop (optional)
    model.eval()
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            # Calculate validation loss and metrics

# Save the trained model
torch.save(model.state_dict(), 'mobilenetv3_fine_tuned.pth')


