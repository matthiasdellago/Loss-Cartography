#simple_cnn.py
"""
Module for a simple CNN model for MNIST classification.
Designed for testing the Cartographer class.
Contains the model, the dataloader, and the training loop.
CNN is a ParameterVector for easy manipulation of the model's parameters.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    """
    A simple convolutional neural network for classifying MNIST digits.
    ParameterVector: Supports vector operations on the model's parameters.
    """
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # First convolutional layer: 1 input channel, 10 output channels, kernel size 5
        # MNIST images are 1x28x28 (grayscale), so no need for more input channels
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        # Second convolutional layer: 10 input channels, 20 output channels, kernel size 5
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        # Fully connected layer: 320 input features (from 20 channels of 4x4 image), 10 output features for 10 classes
        self.fc = nn.Linear(320, 10)

    def forward(self, x):
        # Apply the first convolution, followed by ReLU activation function and max pooling
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # Apply the second convolution, followed by ReLU activation function and max pooling
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        # Flatten the tensor
        x = x.view(-1, 320)
        # Apply the fully connected layer
        return self.fc(x)