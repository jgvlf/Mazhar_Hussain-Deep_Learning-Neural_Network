from .device import device
import torch.nn as nn
import torch.nn.functional as F


# Define Models Architecture
class CNN(nn.Module):
    def __init__(self):

        # Initialize the class as a subclass of nn.Module
        super(CNN, self).__init__()
        self.to(device)

        # First convolutional layer
        self.conv1 = nn.Conv2d(3, 6, 5)

        # First max pooling layer
        self.pool1 = nn.MaxPool2d(2, 2)

        # Second convolutional layer
        self.conv2 = nn.Conv2d(6, 16, 5)

        # Second max pooling layer
        self.pool2 = nn.MaxPool2d(2, 2)

        # 3 Fully connected layers
        # Linear transformation to 120-dimensional space
        self.fc1 = nn.Linear(16 * 5 * 5, 120)

        # Linear transformation to 84-dimensional space
        self.fc2 = nn.Linear(120, 84)

        # Linear transformation to 10-dimensional space (output classes)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Operation 1: First convolutional layer with ReLU activation and max pooling
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)

        # Operation 2: Second convolutional layer with ReLU activation and max pooling
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)

        # Operation 3: Flattened Layer: Reshape for fully connected layer
        x = x.view(-1, 16 * 5 * 5)

        # Operation 4: First fully connected layer with ReLU activation
        x = self.fc1(x)
        x = F.relu(x)

        # Operation 5: Second fully connected layer with ReLU activation
        x = self.fc2(x)
        x = F.relu(x)

        # Operation 6: Output layer (fully connected) with raw scores for each class
        x = self.fc3(x)

        return x

    def get_summary(self):
        from torchsummary import summary
        return summary(self, (3, 32, 32))


