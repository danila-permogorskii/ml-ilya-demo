"""
CNN model architecture for image classification.
Designed to efficiently utilize GPU acceleration.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    """
    A simple convolutional neural network for image classification.
    
    This model can process standard 3-channel RGB images and classify them
    into one of the predefined classes. The architecture is designed to be
    lightweight enough for real-time inference while maintaining good accuracy.
    """
    
    def __init__(self, num_classes=10):
        """
        Initialize the CNN model.
        
        Args:
            num_classes (int): Number of output classes
        """
        super(SimpleCNN, self).__init__()
        
        # First convolutional block
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Second convolutional block
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Third convolutional block
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Calculate size after convolutions and pooling (assuming 32x32 input)
        # 32x32 -> 16x16 -> 8x8 -> 4x4
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, height, width)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes)
        """
        # First block
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        
        # Second block
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        
        # Third block
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    
    def to_gpu(self):
        """
        Move the model to GPU if available.
        
        Returns:
            SimpleCNN: The model on the appropriate device
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return self.to(device)


# Extended model with residual connections for better performance
class ResidualCNN(nn.Module):
    """
    A CNN with residual connections for improved gradient flow and performance.
    
    This model is designed for more complex image classification tasks where
    simple CNNs might struggle. The residual connections help with the vanishing
    gradient problem in deeper networks.
    """
    
    def __init__(self, num_classes=10):
        """
        Initialize the residual CNN model.
        
        Args:
            num_classes (int): Number of output classes
        """
        super(ResidualCNN, self).__init__()
        
        # Initial convolution
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Residual blocks
        self.res_block1 = self._make_res_block(64, 64)
        self.res_block2 = self._make_res_block(64, 128, stride=2)
        self.res_block3 = self._make_res_block(128, 256, stride=2)
        
        # Global average pooling and final classification
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)
        
    def _make_res_block(self, in_channels, out_channels, stride=1):
        """
        Create a residual block.
        
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            stride (int): Stride for the first convolution
            
        Returns:
            nn.Sequential: A residual block
        """
        layers = []
        
        # First convolution with optional downsampling
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        
        # Second convolution
        layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        
        # Shortcut connection
        shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            
        # Add shortcut connection to the main path
        layers.append(shortcut)
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, height, width)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes)
        """
        # Initial convolution
        x = F.relu(self.bn1(self.conv1(x)))
        
        # Residual blocks with shortcuts
        x = F.relu(self.res_block1(x) + x)
        
        # For blocks that change dimensions, the shortcut is within the block
        identity = self.res_block2[:-1](x)
        x = F.relu(identity + self.res_block2[-1](x))
        
        identity = self.res_block3[:-1](x)
        x = F.relu(identity + self.res_block3[-1](x))
        
        # Global pooling and classification
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x
    
    def to_gpu(self):
        """
        Move the model to GPU if available.
        
        Returns:
            ResidualCNN: The model on the appropriate device
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return self.to(device)
