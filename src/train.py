# Imports
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from os import mkdir

from models import DigitClassifier

# Select GPU if CUDA is available otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Convert MNIST images to pytorch tensors
transform = transforms.ToTensor()

# Load the training dataset
train_dataset = datasets.MNIST(
    root="dataset",
    train=True,
    download=True,
    transform=transform
)

# Load the testing dataset
test_dataset = datasets.MNIST(
    root="dataset",
    train=False,
    download=True,
    transform=transform
)

# Divide dataset into batches of 64 images
train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)

# Init nn and move to selected device
model = DigitClassifier().to(device)

# Loss function
criterion = nn.CrossEntropyLoss()

# Optimizer
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

# Number of complete passes through the training dataset
epochs = 5 

# Training loop
for epoch in range(epochs):
    model.train()

    total_loss = 0

    # Process each batch of training images
    for images, lables in train_loader:
        # Move images to CPU or GPU
        images = images.to(device)
        lables = lables.to(device)

        # Clear gradient from previous iterations
        optimizer.zero_grad()

        # Get prediction
        outputs = model(images)

        # Calcute prediction error
        loss = criterion(outputs, lables)

        # Calculate gradients
        loss.backward()

        # Update model weights
        optimizer.step()

        # Accumulate loss
        total_loss += loss.item()

    # Calculate avg training loss for this epoch
    avg_loss = total_loss / len(train_loader)

    # Evaluate model on test dataset
    model.eval()

    correct = 0
    total = 0

    # Disable gradient calculations during evaluation
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Get predictions
            outputs = model(images)

            # Find the class with the highest score
            predictions = outputs.argmax(dim=1)

            # Count correct predictions
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    # Calculate testing accuracy
    accuracy = 100 * correct / total

    print(
        f"Epoch [{epoch + 1}/{epochs}] "
        f"Loss: {avg_loss:.4f} "
        f"Accuracy: {accuracy:.2f}%"
    )

# Save the trained model's parameters
mkdir("checkpoints")
torch.save(model.state_dict(), "checkpoints/model.pth")
print("Model saved successfully!")
