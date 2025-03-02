"""
Utilities module initialization.
"""

from .gpu_utils import get_device_info, optimize_gpu_memory

__all__ = ['get_device_info', 'optimize_gpu_memory']
