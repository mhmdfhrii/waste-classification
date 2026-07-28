import torch
from model import WasteCNN

model = WasteCNN(num_classes=5)

x = torch.randn(1, 3, 224, 224)

y = model(x)

print(y.shape)