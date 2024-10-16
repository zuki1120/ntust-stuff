import torch
from torch import nn, optim
from torchsummary import summary

class regression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(regression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.linear(x)
        return x
    
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = regression(1, 1).to(device)

    summary(model, (1, 1))