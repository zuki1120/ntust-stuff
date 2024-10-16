import torch
from torch import nn
from torchsummary import summary

class vgg(nn.Module):
    def __init__(self, img_size, output):
        super(vgg, self).__init__()
        self.block1 = nn.Sequential(nn.Conv2d(3, 64, 3, 1, 1),
                                    nn.ReLU(),
                                    nn.Conv2d(64, 64, 3, 1, 1),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2, 2)
                                    )
        img_size //= 2

        self.block2 = nn.Sequential(nn.Conv2d(64, 128, 3, 1, 1),
                                    nn.ReLU(),
                                    nn.Conv2d(128, 128, 3, 1, 1),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2, 2)
                                    )
        img_size //= 2

        self.block3 = nn.Sequential(nn.Conv2d(128, 256, 3, 1, 1),
                                    nn.ReLU(),
                                    nn.Conv2d(256, 256, 3, 1, 1),
                                    nn.ReLU(),
                                    nn.Conv2d(256, 256, 3, 1, 1),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2, 2)
                                    )
        img_size //= 2

        self.block4 = nn.Sequential(nn.Conv2d(256, 512, 3, 1, 1),
                                    nn.ReLU(),
                                    nn.Conv2d(512, 512, 3, 1, 1),
                                    nn.ReLU(),
                                    nn.Conv2d(512, 512, 3, 1, 1),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2, 2)
                                    )
        img_size //= 2

        self.block5 = nn.Sequential(nn.Conv2d(512, 512, 3, 1, 1),
                                    nn.ReLU(),
                                    nn.Conv2d(512, 512, 3, 1, 1),
                                    nn.ReLU(),
                                    nn.Conv2d(512, 512, 3, 1, 1),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2, 2)
                                    )
        img_size //= 2

        self.fc1 = nn.Linear(512 * img_size * img_size, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.out = nn.Linear(4096, output)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)


    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.out(x)

        return x
    
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = vgg(32, 10).to(device)

    summary(model, (3, 32, 32))    