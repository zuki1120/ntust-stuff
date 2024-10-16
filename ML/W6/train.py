import torch
import torchvision
import matplotlib.pyplot as plt
from model import CNN
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

train_path = 'dog-cat/train/'
test_path = 'dog-cat/test/'

transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.404), (0.229, 0.224, 0.225))])

train_data = datasets.ImageFolder(train_path, transform = transform)
test_data = datasets.ImageFolder(test_path, transform = transform)

train_loader = DataLoader(train_data, batch_size = 32, shuffle = True)
test_loader = DataLoader(test_data, batch_size = 16, shuffle = False)

# print('label：', train_data.class_to_idx)
# print('path & label：', train_data.imgs[0])
# print('image：')
# print(train_data[0][0])
# print('label：', train_data[0][1])
# print(train_data[0][0].shape)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNN(2).to(device)
loss_f = nn.CrossEntropyLoss()
opt = optim.Adam(model.parameters(), lr = 1e-4)

def train():
    model.train()
    train_loss = 0
    train_acc = 0

    for _, (data, target) in enumerate(train_loader):
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

    for _, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)

        pred = model(data)
        loss = loss_f(pred, target)

        test_loss += loss.item()

    return test_loss / len(test_loader)

if __name__ == '__main__':
    epoch = 3
    train_losses = []
    test_losses = []
    train_acces = []

    for i in range(epoch):
        train_loss, train_acc = train()
        test_loss = test()

        train_losses.append(train_loss)
        train_acces.append(train_acc)
        test_losses.append(test_loss)

        print(f'epoch：{i + 1}, train loss：{train_loss:.6f}, train acc：{train_acc:.3f}, test loss：{test_loss:.6f}')
        if(i + 1) % 5 == 0:
            torch.save(model.state_dict(), f'CNN_{i + 1}.pt')

    plt.figure(1)
    plt.plot(train_losses, label = 'train loss')
    plt.plot(train_acces, label = 'train acc')
    plt.plot(test_losses, label = 'test loss')
    plt.xlabel('epoch')
    plt.ylabel('loss / acc')
    plt.legend()

    plt.savefig('loss.png')
    plt.show()