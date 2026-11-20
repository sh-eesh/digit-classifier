# Imports
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

# Load the dataset
dataset = datasets.MNIST(
    root="dataset",
    train=True,
    download=True,
    transform=transforms.ToTensor()
)

# Change the value to load different images
image, label = dataset[67]

# Show the image and the corresponding label
plt.imshow(image.squeeze(), cmap="gray")
plt.title(f"Label: {label}")
plt.axis("off")
plt.show()
