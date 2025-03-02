"""
GPU utility functions for optimizing GPU usage.
"""

import torch


def get_device_info():
    """
    Get information about available GPU devices.
    
    Returns:
        dict: Information about available devices
    """
    device_info = {
        'cuda_available': torch.cuda.is_available(),
        'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'current_device': None,
        'device_name': None,
        'memory_allocated': None,
        'memory_reserved': None,
        'max_memory_allocated': None,
        'max_memory_reserved': None
    }
    
    if device_info['cuda_available']:
        device_info['current_device'] = torch.cuda.current_device()
        device_info['device_name'] = torch.cuda.get_device_name(device_info['current_device'])
        
        # Get memory information (in GB)
        bytes_to_gb = 1024 ** 3
        device_info['memory_allocated'] = torch.cuda.memory_allocated() / bytes_to_gb
        device_info['memory_reserved'] = torch.cuda.memory_reserved() / bytes_to_gb
        device_info['max_memory_allocated'] = torch.cuda.max_memory_allocated() / bytes_to_gb
        device_info['max_memory_reserved'] = torch.cuda.max_memory_reserved() / bytes_to_gb
        
        # Device capabilities
        device_info['capabilities'] = torch.cuda.get_device_capability(device_info['current_device'])
        
        # Check for tensor cores (available in Volta, Turing, and Ampere architectures)
        major, minor = device_info['capabilities']
        device_info['has_tensor_cores'] = major >= 7
        
    return device_info


def print_device_info():
    """
    Print formatted information about available GPU devices.
    """
    info = get_device_info()
    
    print("===== GPU Information =====")
    
    if not info['cuda_available']:
        print("No CUDA devices available. Using CPU.")
        return
    
    print(f"CUDA Available: {info['cuda_available']}")
    print(f"Number of devices: {info['device_count']}")
    print(f"Current device: {info['current_device']}")
    print(f"Device name: {info['device_name']}")
    print(f"Device capabilities: {info['capabilities']}")
    print(f"Has tensor cores: {info['has_tensor_cores']}")
    
    print("\nMemory Usage:")
    print(f"  Allocated: {info['memory_allocated']:.2f} GB")
    print(f"  Reserved: {info['memory_reserved']:.2f} GB")
    print(f"  Max Allocated: {info['max_memory_allocated']:.2f} GB")
    print(f"  Max Reserved: {info['max_memory_reserved']:.2f} GB")
    
    
def optimize_gpu_memory(enable_tf32=True, enable_cudnn=True, benchmark_mode=True, 
                       deterministic=False, empty_cache=True):
    """
    Optimize GPU memory usage and performance settings.
    
    Args:
        enable_tf32 (bool): Whether to enable TensorFloat-32 (TF32) precision
        enable_cudnn (bool): Whether to enable cuDNN
        benchmark_mode (bool): Whether to enable cuDNN benchmark mode
        deterministic (bool): Whether to enable deterministic mode
        empty_cache (bool): Whether to empty cache before starting
        
    Returns:
        dict: Dictionary with original settings
    """
    original_settings = {}
    
    if torch.cuda.is_available():
        # Store original settings
        if hasattr(torch.backends.cuda, 'matmul'):
            original_settings['tf32_enabled'] = torch.backends.cuda.matmul.allow_tf32
        original_settings['cudnn_enabled'] = torch.backends.cudnn.enabled
        original_settings['benchmark_enabled'] = torch.backends.cudnn.benchmark
        original_settings['deterministic_enabled'] = torch.backends.cudnn.deterministic
        
        # TensorFloat-32 (TF32) settings
        # TF32 is a math mode introduced in NVIDIA's Ampere architecture 
        # that speeds up operations by rounding float32 to a lower precision format
        if hasattr(torch.backends.cuda, 'matmul'):
            torch.backends.cuda.matmul.allow_tf32 = enable_tf32
            torch.backends.cudnn.allow_tf32 = enable_tf32
            
        # cuDNN settings
        torch.backends.cudnn.enabled = enable_cudnn
        torch.backends.cudnn.benchmark = benchmark_mode
        torch.backends.cudnn.deterministic = deterministic
        
        # Empty cache if requested
        if empty_cache:
            torch.cuda.empty_cache()
            
    return original_settings


def allocate_gpu_memory(fraction=0.8):
    """
    Pre-allocate a fraction of GPU memory to avoid fragmentation.
    
    This can help with memory fragmentation issues by allocating 
    a large chunk of memory at the start and then releasing it.
    
    Args:
        fraction (float): Fraction of GPU memory to allocate (0.0-1.0)
        
    Returns:
        torch.Tensor: The allocated tensor (keep a reference to control when it's freed)
    """
    if not torch.cuda.is_available():
        print("No CUDA device available. Memory allocation skipped.")
        return None
    
    # Get total memory
    total_memory = torch.cuda.get_device_properties(0).total_memory
    target_memory = int(total_memory * fraction)
    
    try:
        # Allocate memory
        print(f"Pre-allocating {fraction * 100:.1f}% of GPU memory...")
        x = torch.empty(target_memory, dtype=torch.int8, device='cuda')
        print("Memory allocated successfully.")
        return x
    except RuntimeError as e:
        print(f"Memory allocation failed: {e}")
        return None


def distributed_setup(rank, world_size):
    """
    Set up distributed training.
    
    Args:
        rank (int): Rank of the current process
        world_size (int): Number of processes
    """
    torch.distributed.init_process_group(
        backend='nccl',  # NCCL is the best backend for GPU training
        init_method='tcp://127.0.0.1:12355',
        world_size=world_size,
        rank=rank
    )
    
    # Set the device for this process
    torch.cuda.set_device(rank)
    

def cleanup_distributed():
    """
    Clean up distributed training resources.
    """
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
