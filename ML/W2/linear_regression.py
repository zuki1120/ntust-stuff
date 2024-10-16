import torch
import matplotlib.pyplot as plt
from torch import nn, optim
import torch.cuda
from model import regression

x = torch.randn(100, 1)
w = torch.tensor([10.])
b = torch.tensor([3.])
y = w * x + b + torch.randn(x.shape) * 2.5

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = regression(1, 1).to(device)
cirterion = nn.MSELoss()
opt = optim.SGD(model.parameters(), lr = 0.1)

epoch = 100
losses = []

for i in range(epoch):
    input, target = x.to(device), y.to(device)
    model.train()

    opt.zero_grad()

    pred = model(input)
    loss = cirterion(target, pred)

    loss.backward()
    opt.step()

    losses.append(loss.item())

    print('epoch {}, loss {}'.format(i, loss))

pred = model(x.to(device)).cpu().detach()

plt.figure(1)
plt.plot(x, y, 'ro')
plt.plot(x, pred, 'b-')
plt.xlabel('x')
plt.ylabel('y')
plt.title('data')
plt.savefig('data.png')

plt.figure(2)
plt.plot(losses, label = 'loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('loss')
plt.legend()
plt.savefig('loss.png')
plt.show()