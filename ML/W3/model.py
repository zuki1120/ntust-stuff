import torch
from torch import nn
import torch.cuda
from torchsummary import summary

class regression(nn.Module):
    def __init__(self, input_dim):
        super(regression, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        x = self.linear(x)

        return x
    
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = regression(87).to(device)

    summary(model, (1, 87))
