import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from model import vgg
from torch import nn, optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

batch_size = 64
lr = 1e-4

transform = transforms.Compose((transforms.ToTensor(),
                                transforms.Normalize((0.465, 0.456, 0.406), (0.229, 0.224, 0.225))))
train_data = datasets.CIFAR10('data', train = True, transform = transform, download = True)
test_data = datasets.CIFAR10('data', train = False, transform = transform, download = True)

train_loader = DataLoader(train_data, batch_size = batch_size, shuffle = True)
test_loader = DataLoader(test_data, batch_size = batch_size, shuffle = False)
cls = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# plt.imshow(train_data.data[0])
# plt.title(np.array(cls)[train_data.targets[0]])
# plt.show()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = vgg(32, 10).to(device)
loss_f = nn.CrossEntropyLoss()
opt = optim.Adam(model.parameters(), lr = lr)

def train():
    model.train()
    train_loss = 0
    train_acc = 0
    for _, (data, target) in tqdm(enumerate(train_loader), total = len(train_loader), leave = True):
        data, target = data.to(device), target.to(device)

        opt.zero_grad()
        pred = model(data)
        loss = loss_f(pred, target)
        loss.backward()
        opt.step()

        train_loss += loss.item()
        _, y = pred.max(1)
        correct = (y == target).sum().item()
        acc = correct / data.shape[0]
        train_acc += acc

    return train_loss / len(train_loader), train_acc / len(train_loader)

def test():
    model.eval()
    test_loss = 0
    test_acc = 0
    for _, (data, target) in tqdm(enumerate(test_loader), total = len(test_loader), leave = True):
        data, target = data.to(device), target.to(device)

        pred = model(data)
        loss = loss_f(pred, target)

        test_loss += loss.item()
        _, y = pred.max(1)
        correct = (y == target).sum().item()
        acc = correct / data.shape[0]
        test_acc += acc

    return test_loss / len(test_loader), test_acc / len(test_loader)

if __name__ == '__main__':
    epoch = 10
    train_losses = []
    train_acces = []
    test_losses = []
    test_acces = []

    for i in range(epoch):
        train_loss, train_acc = train()
        test_loss, test_acc = test()

        train_losses.append(train_loss)
        train_acces.append(train_acc)
        test_losses.append(test_loss)
        test_acces.append(test_acc)

        print(f'epoch: {i + 1}, train loss: {train_loss:.6f},train acc: {train_acc:.2f}, test loss: {test_loss:.6f}, test acc: {test_acc:.2f}')

        if (i + 1) % 5 == 0:
            torch.save(model.state_dict(), f'vgg_{i + 1}.pt')

    plt.figure(1)
    plt.plot(train_losses, label = 'train loss')
    plt.plot(train_acces, label = 'train acc')
    plt.plot(test_losses, label = 'test loss')
    plt.plot(test_acces, label = 'test acc')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss / acc')
    plt.savefig('cifar_loss.png')
    plt.show()
