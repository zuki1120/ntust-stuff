import torch
import torchvision
import matplotlib.pyplot as plt
from mlp import MLP
from torchvision import transforms
from torchvision.datasets import mnist

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307, ), (0.3081, ))])
data = mnist.MNIST('data/mnist', train = False, transform = transform, download = False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MLP(784, 10).to(device)

model.load_state_dict(torch.load('MLP.pt'))
model.eval()

test_data = torch.unsqueeze(data.test_data, dim = 1).type(torch.FloatTensor)[:2000] / 255.
pred = model(test_data)
pred = pred.data.cpu().numpy()
y = pred.max(1)

im = data.data[0].cpu().numpy()
plt.figure()
plt.imshow(im, cmap = 'gray')
plt.title(y[0])
plt.savefig('pred.png')
plt.show()