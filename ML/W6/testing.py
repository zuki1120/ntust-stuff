import torch
from PIL import Image
import matplotlib.pyplot as plt
import torch.cuda
from torchvision import transforms
from model import CNN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transform = transforms.Compose((transforms.Resize((224, 224)), transforms.ToTensor(),
                                 transforms.Normalize((0.465, 0.456, 0.406), (0.229, 0.224, 0.225))))

model = CNN(2).to(device)
model.load_state_dict(torch.load('CNN10.pt', map_location = 'cpu'))

img = Image.open('test.jpg').convert('RGB')
data = transform(img)
data = torch.unsqueeze(data, dim = 0).to(device)

pred = model(data)
_, y = pred.max(1)

plt.figure(1)
plt.imshow(img)
plt.title('dog' if(y.cpu().numpy() == 0) else 'cat')
plt.xticks([])
plt.yticks([])
plt.savefig('test.png')
plt.show()