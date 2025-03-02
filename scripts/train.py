#!/usr/bin/env python
"""
Training script for the ML-Ilya-Demo project.

This script handles model training using command line arguments.
"""

import os
import argparse
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

# Import from package
from src.models import SimpleCNN, ResidualCNN
from src.data import get_cifar10_loaders
from src.training import Trainer
from src.utils import get_device_info, optimize_gpu_memory


def parse_args():
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Train image classification model')
    
    # Model parameters
    parser.add_argument('--model', type=str, default='simple', choices=['simple', 'residual'],
                        help='Model architecture (simple or residual)')
    parser.add_argument('--num_classes', type=int, default=10,
                        help='Number of output classes')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay (L2 penalty)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    
    # Optimizer parameters
    parser.add_argument('--optimizer', type=str, default='adam', 
                        choices=['adam', 'sgd', 'adamw'],
                        help='Optimizer type')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='Momentum for SGD optimizer')
    parser.add_argument('--scheduler', action='store_true',
                        help='Use learning rate scheduler')
    
    # Hardware parameters
    parser.add_argument('--cuda', action='store_true',
                        help='Use CUDA if available')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='ID of GPU to use')
    parser.add_argument('--deterministic', action='store_true',
                        help='Make training deterministic')
    
    # Storage parameters
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    return parser.parse_args()


def setup_environment(args):
    """
    Set up the training environment.
    
    Args:
        args (argparse.Namespace): Parsed arguments
        
    Returns:
        torch.device: Device to use for training
    """
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    
    # Optimize GPU settings
    if args.cuda and torch.cuda.is_available():
        optimize_gpu_memory(
            enable_tf32=True,
            enable_cudnn=True,
            benchmark_mode=not args.deterministic,
            deterministic=args.deterministic,
            empty_cache=True
        )
        
        device = torch.device(f'cuda:{args.gpu_id}')
        print(f"Using GPU: {torch.cuda.get_device_name(args.gpu_id)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
        
    # Print device information
    if device.type == 'cuda':
        device_info = get_device_info()
        print(f"Number of CUDA devices: {device_info['device_count']}")
        print(f"Current CUDA device: {device_info['device_name']}")
        print(f"CUDA capabilities: {device_info['capabilities']}")
        print(f"Has tensor cores: {device_info['has_tensor_cores']}")
        
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
        
    return device


def create_model(args, device):
    """
    Create and initialize the model.
    
    Args:
        args (argparse.Namespace): Parsed arguments
        device (torch.device): Device to use
        
    Returns:
        nn.Module: Initialized model
    """
    if args.model == 'simple':
        model = SimpleCNN(num_classes=args.num_classes)
    else:  # residual
        model = ResidualCNN(num_classes=args.num_classes)
        
    model = model.to(device)
    return model


def create_optimizer(args, model_parameters):
    """
    Create optimizer based on arguments.
    
    Args:
        args (argparse.Namespace): Parsed arguments
        model_parameters: Model parameters to optimize
        
    Returns:
        optim.Optimizer: Initialized optimizer
    """
    if args.optimizer == 'adam':
        optimizer = optim.Adam(
            model_parameters, 
            lr=args.lr, 
            weight_decay=args.weight_decay
        )
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(
            model_parameters, 
            lr=args.lr, 
            weight_decay=args.weight_decay
        )
    else:  # sgd
        optimizer = optim.SGD(
            model_parameters, 
            lr=args.lr, 
            momentum=args.momentum, 
            weight_decay=args.weight_decay
        )
        
    return optimizer


def main():
    """
    Main training function.
    """
    # Parse arguments
    args = parse_args()
    
    # Set up environment
    device = setup_environment(args)
    
    # Load data
    print("Loading dataset...")
    train_loader, val_loader, classes = get_cifar10_loaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    print(f"Dataset loaded. {len(train_loader.dataset)} training samples, " 
          f"{len(val_loader.dataset)} validation samples")
    print(f"Classes: {classes}")
    
    # Create model
    print(f"Creating {args.model} model...")
    model = create_model(args, device)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create optimizer
    optimizer = create_optimizer(args, model.parameters())
    
    # Create scheduler if requested
    scheduler = None
    if args.scheduler:
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        checkpoint_dir=args.checkpoint_dir
    )
    
    # Train model
    print(f"Starting training for {args.epochs} epochs...")
    history = trainer.train(
        epochs=args.epochs,
        save_best=True,
        early_stopping=5  # Stop if no improvement for 5 epochs
    )
    
    # Plot training history
    trainer.plot_history()
    
    print("Training completed successfully!")
    

if __name__ == "__main__":
    main()
