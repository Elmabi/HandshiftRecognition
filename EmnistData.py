import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import datasets, transforms
import numpy as np

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)), ])

train_set = datasets.EMNIST('Data2FürPytorch', split="balanced", download=True, train=True, transform=transform)
test_set = datasets.EMNIST('Data2FürPytorch', split="balanced", download=True, train=False, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=True)

images, labels = next(iter(train_loader))

# print(images.shape)
# print(labels.shape)
# print(len(test_set)) # How many image in the test_set
print(train_set.targets.bincount())  # Datasets is balanced
# print(train_set.targets) # label tensor for the data, the value represent element
# print(plt.imshow(images[0].reshape(28, 28), cmap="gray"))
# print(plt.imshow(images.numpy().squeeze(), cmap='gray_r'))
# print('label: ', labels)
# batch = next(iter(test_loader))
# print(len(batch))
# print(type(batch))
# images, labels = batch
# print(images.shape)
# grid = torchvision.utils.make_grid(images, nrow=10)
# print(plt.figure(figsize=(15,15)))
# print(plt.imshow(np.transpose(grid, (1,2,0))))
# print("labels: ", labels)
