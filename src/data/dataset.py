"""
Dataset handling utilities for image classification.
"""

import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms


def get_cifar10_loaders(batch_size=64, num_workers=4, augment=True, download=True):
    """
    Get DataLoaders for the CIFAR-10 dataset.
    
    Args:
        batch_size (int): Batch size for training and testing
        num_workers (int): Number of worker processes for data loading
        augment (bool): Whether to apply data augmentation
        download (bool): Whether to download the dataset if not available
        
    Returns:
        tuple: (train_loader, test_loader, classes)
    """
    # Define normalization parameters for CIFAR-10
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)
    
    # Basic transformations for validation/test
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    # Apply additional augmentations for training if requested
    if augment:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        train_transform = test_transform
    
    # Load training data
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', 
        train=True,
        download=download,
        transform=train_transform
    )
    
    # Load test data
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', 
        train=False,
        download=download,
        transform=test_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()  # Pin memory if using CUDA
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size,
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    # CIFAR-10 classes
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck')
    
    return train_loader, test_loader, classes


def create_data_loaders_from_custom_dataset(train_dir, test_dir, batch_size=64, 
                                          img_size=224, num_workers=4):
    """
    Create DataLoaders from custom image directories.
    
    Args:
        train_dir (str): Path to training data directory
        test_dir (str): Path to test data directory
        batch_size (int): Batch size for training and testing
        img_size (int): Size to resize images to
        num_workers (int): Number of worker processes for data loading
        
    Returns:
        tuple: (train_loader, test_loader)
    """
    # Define transformations
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = torchvision.datasets.ImageFolder(
        root=train_dir,
        transform=train_transform
    )
    
    test_dataset = torchvision.datasets.ImageFolder(
        root=test_dir,
        transform=test_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size,
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, test_loader, train_dataset.classes
