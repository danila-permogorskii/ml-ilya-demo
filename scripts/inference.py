#!/usr/bin/env python
"""
Inference script for the ML-Ilya-Demo project.

This script handles model inference for image classification on a trained model.
"""

import os
import argparse
import time
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

# Import from package
from src.models import SimpleCNN, ResidualCNN
from src.utils import get_device_info, optimize_gpu_memory


def parse_args():
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Run inference with a trained model')
    
    # Model parameters
    parser.add_argument('--model_type', type=str, default='simple', choices=['simple', 'residual'],
                        help='Model architecture (simple or residual)')
    parser.add_argument('--num_classes', type=int, default=10,
                        help='Number of output classes')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained model checkpoint')
    
    # Input parameters
    parser.add_argument('--image_path', type=str, required=True,
                        help='Path to input image')
    parser.add_argument('--batch_mode', action='store_true',
                        help='Process a directory of images')
    
    # Output parameters
    parser.add_argument('--class_names', type=str, default='plane,car,bird,cat,deer,dog,frog,horse,ship,truck',
                        help='Comma-separated list of class names')
    parser.add_argument('--top_k', type=int, default=5,
                        help='Show top K predictions')
                        
    # Hardware parameters
    parser.add_argument('--cuda', action='store_true',
                        help='Use CUDA if available')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='ID of GPU to use')
    parser.add_argument('--half_precision', action='store_true',
                        help='Use half precision (FP16)')
    
    # Performance parameters
    parser.add_argument('--benchmark', action='store_true',
                        help='Run inference benchmark')
    parser.add_argument('--num_runs', type=int, default=100,
                        help='Number of runs for benchmarking')
    
    return parser.parse_args()


def setup_environment(args):
    """
    Set up the inference environment.
    
    Args:
        args (argparse.Namespace): Parsed arguments
        
    Returns:
        torch.device: Device to use for inference
    """
    # Optimize GPU settings
    if args.cuda and torch.cuda.is_available():
        optimize_gpu_memory(
            enable_tf32=True,
            enable_cudnn=True,
            benchmark_mode=args.benchmark,
            deterministic=False,
            empty_cache=True
        )
        
        device = torch.device(f'cuda:{args.gpu_id}')
        print(f"Using GPU: {torch.cuda.get_device_name(args.gpu_id)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
        
    # Print device information if GPU
    if device.type == 'cuda':
        device_info = get_device_info()
        print(f"CUDA Device: {device_info['device_name']}")
        
    return device


def load_model(args, device):
    """
    Load the trained model from checkpoint.
    
    Args:
        args (argparse.Namespace): Parsed arguments
        device (torch.device): Device to use
        
    Returns:
        nn.Module: Loaded model
    """
    # Create the model architecture
    if args.model_type == 'simple':
        model = SimpleCNN(num_classes=args.num_classes)
    else:  # residual
        model = ResidualCNN(num_classes=args.num_classes)
    
    # Load checkpoint
    checkpoint = torch.load(args.model_path, map_location=device)
    
    # Load state dictionary (handle different formats)
    if 'model_state_dict' in checkpoint:
        # Our trainer format
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # Direct state dict format
        model.load_state_dict(checkpoint)
    
    # Move model to device and set to evaluation mode
    model = model.to(device)
    model.eval()
    
    # Use half precision if requested
    if args.half_precision and device.type == 'cuda':
        model = model.half()
        print("Using half precision (FP16)")
    
    return model


def preprocess_image(image_path):
    """
    Preprocess an image for inference.
    
    Args:
        image_path (str): Path to the image
        
    Returns:
        torch.Tensor: Preprocessed image tensor
    """
    # Define normalization parameters (same as training)
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)
    
    # Define transform
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    # Load and transform image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    
    return image_tensor


def run_inference(model, image_tensor, device, half_precision=False):
    """
    Run inference on an input image.
    
    Args:
        model (nn.Module): Trained model
        image_tensor (torch.Tensor): Preprocessed image tensor
        device (torch.device): Device to use
        half_precision (bool): Whether to use half precision
        
    Returns:
        tuple: (probabilities, class_indices)
    """
    # Move input to device
    image_tensor = image_tensor.to(device)
    
    # Convert to half precision if requested
    if half_precision and device.type == 'cuda':
        image_tensor = image_tensor.half()
    
    # Run inference
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = F.softmax(output, dim=1)[0]
        
    # Get sorted indices and probabilities
    probs, indices = torch.sort(probabilities, descending=True)
    
    return probs.cpu().numpy(), indices.cpu().numpy()


def benchmark_inference(model, image_tensor, device, half_precision=False, num_runs=100):
    """
    Benchmark inference performance.
    
    Args:
        model (nn.Module): Trained model
        image_tensor (torch.Tensor): Preprocessed image tensor
        device (torch.device): Device to use
        half_precision (bool): Whether to use half precision
        num_runs (int): Number of inference runs
        
    Returns:
        dict: Benchmark results
    """
    # Move input to device
    image_tensor = image_tensor.to(device)
    
    # Convert to half precision if requested
    if half_precision and device.type == 'cuda':
        image_tensor = image_tensor.half()
    
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model(image_tensor)
    
    # Synchronize before timing
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Benchmark
    timings = []
    with torch.no_grad():
        for _ in range(num_runs):
            if device.type == 'cuda':
                torch.cuda.synchronize()
                
            start_time = time.time()
            _ = model(image_tensor)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
                
            end_time = time.time()
            timings.append((end_time - start_time) * 1000)  # ms
    
    # Calculate statistics
    timings = np.array(timings)
    mean_time = np.mean(timings)
    std_time = np.std(timings)
    median_time = np.median(timings)
    min_time = np.min(timings)
    max_time = np.max(timings)
    
    return {
        'mean_ms': mean_time,
        'std_ms': std_time,
        'median_ms': median_time,
        'min_ms': min_time,
        'max_ms': max_time,
        'fps': 1000 / mean_time
    }


def main():
    """
    Main inference function.
    """
    # Parse arguments
    args = parse_args()
    
    # Setup environment
    device = setup_environment(args)
    
    # Parse class names
    class_names = args.class_names.split(',')
    
    # Load model
    print("Loading model...")
    model = load_model(args, device)
    print("Model loaded successfully")
    
    if args.batch_mode:
        # Batch mode - process all images in the directory
        image_dir = args.image_path
        image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        print(f"Found {len(image_files)} images to process")
        
        for image_file in image_files:
            print(f"\nProcessing: {os.path.basename(image_file)}")
            image_tensor = preprocess_image(image_file)
            probs, indices = run_inference(model, image_tensor, device, args.half_precision)
            
            # Display top-k predictions
            for i in range(min(args.top_k, len(indices))):
                class_idx = indices[i]
                class_name = class_names[class_idx] if class_idx < len(class_names) else f"Class {class_idx}"
                print(f"{class_name}: {probs[i]*100:.2f}%")
    else:
        # Single image mode
        print(f"Processing image: {args.image_path}")
        image_tensor = preprocess_image(args.image_path)
        
        # Benchmark if requested
        if args.benchmark:
            print(f"Running benchmark with {args.num_runs} iterations...")
            benchmark_results = benchmark_inference(
                model, image_tensor, device, 
                args.half_precision, args.num_runs
            )
            
            print("\nBenchmark Results:")
            print(f"Mean inference time: {benchmark_results['mean_ms']:.2f} ms")
            print(f"Median inference time: {benchmark_results['median_ms']:.2f} ms")
            print(f"Std dev: {benchmark_results['std_ms']:.2f} ms")
            print(f"Min: {benchmark_results['min_ms']:.2f} ms, Max: {benchmark_results['max_ms']:.2f} ms")
            print(f"Throughput: {benchmark_results['fps']:.2f} FPS")
        
        # Run actual inference
        probs, indices = run_inference(model, image_tensor, device, args.half_precision)
        
        # Display top-k predictions
        print("\nPredictions:")
        for i in range(min(args.top_k, len(indices))):
            class_idx = indices[i]
            class_name = class_names[class_idx] if class_idx < len(class_names) else f"Class {class_idx}"
            print(f"{class_name}: {probs[i]*100:.2f}%")


if __name__ == "__main__":
    main()
