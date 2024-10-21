import os
import torch
import torchvision
import matplotlib.pyplot as plt
from model import CNN, VGG16
from tqdm import tqdm
from torch import nn, optim
from early_stopping import EarlyStopping
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import datasets, transforms

# Data preprocessing
picture_size = 48
folder_path = './images/'

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((picture_size, picture_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

train_dataset = ImageFolder(root=os.path.join(folder_path, 'train'), transform = transform)
valid_dataset = ImageFolder(root=os.path.join(folder_path, 'validation'), transform = transform)

batch_size = 128

train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
valid_loader = DataLoader(valid_dataset, batch_size = batch_size, shuffle = False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model = VGG16().to(device)
loss_f = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 1e-4)

def train(model, train_loader, valid_loader, criterion, optimizer, epochs = 50):
    model.train()
    train_loss, val_loss = [], []
    train_acc, val_acc = [], []
    # train_loss, train_acc = [], []

    early_stopping = EarlyStopping(patience=3, verbose=True)

    for epoch in range(epochs):
        print(f"Epoch {epoch+1} / {epochs}")
        print('learning rate:', optimizer.param_groups[0]['lr'])
        
        # Training phase
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_loss.append(running_loss / len(train_loader))
        train_acc.append(100 * correct / total)
        
        # Validation phase
        model.eval()
        running_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in tqdm(valid_loader):
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_loss.append(running_loss / len(valid_loader))
        val_acc.append(100 * correct / total)
        
        print(f"Train Loss: {train_loss[-1]:.4f}, Train Acc: {train_acc[-1]:.2f}%")
        print(f"Val Loss: {val_loss[-1]:.4f}, Val Acc: {val_acc[-1]:.2f}%")
        
        early_stopping(val_loss[-1], model, optimizer)
        
        if early_stopping.early_stop:
            print("Early stopping")
            break

        model.train()

    return train_loss, val_loss, train_acc, val_acc

if __name__ == '__main__':
    epochs = 50
    train_loss, val_loss, train_acc, val_acc = train(model, train_loader, valid_loader, loss_f, optimizer, epochs)

    # Plot loss and accuracy
    plt.figure(figsize=(20, 10))

    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('train_plot.png')
    plt.show()