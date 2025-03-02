"""
ML-Ilya-Demo package initialization.
"""

import os
import sys

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import main modules
from src.models import SimpleCNN, ResidualCNN
from src.data import get_cifar10_loaders
from src.training import Trainer
from src.utils import get_device_info, optimize_gpu_memory

__all__ = [
    'SimpleCNN', 
    'ResidualCNN',
    'get_cifar10_loaders',
    'Trainer',
    'get_device_info',
    'optimize_gpu_memory'
]

__version__ = '0.1.0'
