import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets
import numpy as np
import matplotlib.pyplot as plt

# Define transform for training data
train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.RandomResizedCrop(48, scale=(0.8, 1.2)),
    transforms.RandomApply([transforms.RandomAffine(0, translate=(0.2, 0.2))], p=0.5),
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply([transforms.RandomRotation(10)], p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Define transform for validation and test data
val_test_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])


# Create datasets with different transforms
train_dataset = torchvision.datasets.ImageFolder(root=train_directory, transform=train_transform)
val_dataset = torchvision.datasets.ImageFolder(root=val_directory, transform=val_test_transform)
test_dataset = torchvision.datasets.ImageFolder(root=test_directory, transform=val_test_transform)

# Define batch size
batch_size = 120

# Create data loaders for training, validation, and test sets
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Get class names
expressions_list = train_dataset.classes

# Create an iterator
data_iterator = iter(train_loader)

# Get a batch of data
images, labels = next(data_iterator)

print(f"Each batch has {images.size(0)} parts of data.")
print(f"Each batch's images part has the shape of {images.shape}")
print(f"Each batch's labels part has the shape of {labels.shape}")

title_font = {"color": "k", "weight": "bold", "size": 14}

print("Facial expressions are:")

for idx, expression in enumerate(expressions_list):
    print(f"    {idx}. {expression}")

# Generate random indexes for visualization
indexes = np.random.randint(0, len(train_dataset), 14)

fig, axes = plt.subplots(2, 7, figsize=(28, 8))

i = 0
j = 0

for index in indexes:
    img, label = train_dataset[index]
    axes[i, j].imshow(img.squeeze().numpy(), cmap="gray")
    axes[i, j].set_title(label, fontdict=title_font)

    j += 1

    if j == 7:
        i = 1
        j = 0
plt.show()
