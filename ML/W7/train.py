import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from model import vgg
from torch import nn, optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

batch_size = 64
lr = 1e-3

transform = transforms.Compose((transforms.ToTensor(),
                                 transforms.Normalize((0.465, 0.456, 0.406), (0.229, 0.224, 0.225))))
train_data = datasets.CIFAR10('data', train = True, transform = transform, download = True)
test_data = datasets.CIFAR10('data', train = False, transform = transform, download = True)

train_loader = DataLoader(train_data, batch_size = batch_size, shuffle = True)
test_loader = DataLoader(test_data, batch_size = batch_size, shuffle = False)
cls = ('plance', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# plt.imshow(train_data.data[0])
# plt.title(np.array(cls)[train_data.targets[0]])
# plt.show()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = vgg(32, 10).to(device)
loss_f = nn.CrossEntropyLoss()
opt = optim.Adam(model.parameters(), lr = lr)