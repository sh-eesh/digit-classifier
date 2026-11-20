# Imports
import torch.nn as nn

# Define digit classification nn
class DigitClassifier(nn.Module):
    def __init__(self):
        # Init parent pytorch module
        super().__init__()

        # Define our network
        self.network = nn.Sequential(
            # Convert image to a vector
            nn.Flatten(),

            # First layer 784 inputs -> 128 neurons
            nn.Linear(784, 128),

            # Add non linearity
            nn.ReLU(),

            # Second layer 128 inputs -> 64 neurons
            nn.Linear(128, 64),
            nn.ReLU(),

            # Output layer 64 inputs -> 10 digit classes
            nn.Linear(64, 10)
        )

    # Define how input data passes through the network
    def forward(self, x):
        return self.network(x)
