import torch
import torch.nn as nn
import torch.nn.functional as F

# class DQN(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super(DQN, self).__init__()
#         self.fc1 = nn.Linear(input_dim, 128)
#         self.fc2 = nn.Linear(128, 128)
#         self.fc3 = nn.Linear(128, output_dim)

#     def forward(self, x):
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x))
#         return self.fc3(x)

class CNN(nn.Module):
    def __init__(self, size, output_dim):
        super(CNN, self).__init__()
        self.size = size
        self.output_dim = output_dim
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # Calculate the output size of the convolutional layers
        self.conv_output_size = self._get_conv_output_size()
        # Fully connected layers
        self.fc1 = nn.Linear(self.conv_output_size, 128)
        self.fc2 = nn.Linear(128, output_dim)

    def _get_conv_output_size(self):
        # Helper function to calculate the output size after convolution
        # Create a dummy input to pass through the convolutional layers
        dummy_input = torch.zeros(1, 1, self.size, self.size)
        # Pass the dummy input through the convolutional layers
        x = F.relu(self.conv1(dummy_input))
        x = F.relu(self.conv2(x))
        # Flatten the output
        return x.view(1, -1).size(1)

    def forward(self, x):
        # Pass the input through the convolutional layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        # Flatten the output
        x = x.view(-1, self.conv_output_size)
        # Pass the flattened output through the fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
