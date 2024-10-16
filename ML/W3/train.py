import torch
import torchvision
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from torch import nn, optim
from model import regression
from read_data import covid19Data
from torch.utils.data import DataLoader, random_split

batch_size = 8
lr = 1e-3

df = pd.read_csv('Data/covid_train.csv', header = 0)
df = df.to_numpy(dtype = np.float32)
df = torch.from_numpy(df)
data = covid19Data(df)

train_len = int(len(data) // 3)
test_len = len(data) - train_len * 2

torch.manual_seed(0)
train_data, val_data, test_data = random_split(data, [train_len, train_len, test_len])
# print(train_data[0])

train_loader = DataLoader(train_data, batch_size = batch_size, shuffle = True)
val_loader = DataLoader(val_data, batch_size = batch_size, shuffle = True)
test_loader = DataLoader(test_data, batch_size = batch_size, shuffle = False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = regression(87).to(device)
loss_f = nn.MSELoss()
opt = optim.Adagrad(model.parameters(), lr = lr)

def train(epoch):
    model.train()
    losses = 0

    for _, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        opt.zero_grad()

        pred = model(data)
        loss = loss_f(pred, target)
        losses += loss.item()

        loss.backward()
        opt.step()

        if (epoch + 1) % 100 == 0:
            torch.save(model.state_dict(), f'pt/regression_{epoch}.pt')

    print(f'train epoch: {epoch}, train loss: {losses/len(train_loader):.6f}')

    return losses/len(train_loader)

def val(epoch):
    model.eval()
    losses = 0

    for _, (data, target) in enumerate(val_loader):
        data, target = data.to(device), target.to(device)

        pred = model(data)
        loss = loss_f(pred, target)
        losses += loss.item()

    print(f'val epoch: {epoch}, val loss: {losses/len(train_loader):.6f}')

    return losses/len(val_loader)

def test():
    model.eval()
    acc = 0
    total = 0
    with torch.no_grad():
        for data, target in test_data:
            data, target = data.to(device), target.to(device)
            pred = model(data).item()
            error = pred - target.item()
            acc += math.pow(error, 2)
            total += target.size(0)
        mse = math.sqrt(acc / total)

    return mse

if __name__ == '__main__':
    epoch = 1000
    train_losses = []
    val_losses = []

    for i in range(epoch):
        train_loss = train(i)
        val_loss = val(i)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

    mse = test()
    print(mse)

    plt.figure(1)
    plt.plot(train_losses, label = 'train_loss')
    plt.plot(val_losses, label = 'val_loss')

    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig('loss.png')
    plt.show()
    