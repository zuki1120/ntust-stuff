import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from model import vgg
from PIL import Image
from torchvision import datasets, transforms

transform = transforms.Compose([transforms.Resize(32),
                                transforms.ToTensor(),
                                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
test_data = datasets.CIFAR10('data', train = False, transform = transform, download = False)

classes = np.array(('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = vgg(32, 10).to(device)
model.load_state_dict(torch.load('vgg_10.pt'))

# img, target = test_data[0]
img = Image.open('test.jpg').convert('RGB')
data = transform(img)
data = torch.unsqueeze(data, dim = 0).to(device)
pred = model(data)
_, y = pred.max(1)

# img = img.cpu().numpy().transpose((1, 2, 0))

plt.imshow(img)
plt.title(f'{classes[y.cpu().numpy()]}')
plt.savefig('result.png')
plt.show()