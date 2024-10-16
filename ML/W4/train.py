import torch
import torchvision
import matplotlib.pyplot as plt
from mlp import MLP
from torch import nn, optim
from torchvision import transforms
from torchvision.datasets import mnist
from torch.utils.data import DataLoader

batch_size = 128
lr = 1e-4

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307, ), (0.3081, ))])

train_data = mnist.MNIST('data/mnist', train = True, transform = transform, download = True)
test_data = mnist.MNIST('data/mnist', train = False, transform = transform, download = True)

train_loader = DataLoader(train_data, batch_size = batch_size, shuffle = True)
test_loader = DataLoader(test_data, batch_size = batch_size, shuffle = False)

# print(train_data.data[0])
# print(train_data.targets[0])
# im = train_data.data[0].cpu().numpy()

# plt.figure(1)
# plt.imshow(im, cmap = 'gray')
# plt.title(train_data.targets[0].item())
# plt.show()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MLP(784, 10).to(device)
loss_f = nn.CrossEntropyLoss()
opt = optim.Adam(model.parameters(), lr = lr)

def train():
    train_loss = 0
    train_acc = 0
    model.train()

    for _, (im, target) in enumerate(train_loader):
        im, target = im.to(device), target.to(device)

        opt.zero_grad()
        pred = model(im)
        loss = loss_f(pred, target)
        loss.backward()
        opt.step()

        train_loss += loss.item()
        _, y = pred.max(1)
        corrent = (y == target).sum().item()
        acc = corrent / im.shape[0]
        train_acc += acc

    return train_loss / len(train_loader), train_acc / len(train_loader)

def test():
    test_loss = 0
    test_acc = 0
    model.eval()

    for _, (im, target) in enumerate(test_loader):
        im, target = im.to(device), target.to(device)

        pred = model(im)
        loss = loss_f(pred, target)
        test_loss += loss.item()
        _, y = pred.max(1)
        corrent = (y == target).sum().item()
        acc = corrent / im.shape[0]
        test_acc += acc

    return test_loss / len(test_loader), test_acc / len(test_loader)

if __name__ == '__main__':
    epoch = 10
    train_losses = []
    test_losses = []
    train_acces = []
    test_acces = []

    for i in range(epoch):
        train_loss, train_acc = train()
        test_loss, test_acc = test()

        train_losses.append(train_loss)
        train_acces.append(train_acc)
        test_losses.append(test_loss)
        test_acces.append(test_acc)

        print(f'Epoch：{i}, train loss：{train_loss:.6f}, train_acc：{train_acc:.2f}, test loss：{test_loss:.6f}, test_acc：{test_acc:.2f}')

    torch.save(model.state_dict(), 'MLP.pt')

    plt.figure(1)
    plt.plot(train_losses, label = 'train loss')
    plt.plot(train_acces, label = 'train acc')
    plt.plot(test_losses, label = 'test loss')
    plt.plot(test_acces, label = 'test acc')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss / acc')
    plt.savefig('result.png')
    plt.show()