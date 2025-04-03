import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from imports import *

# GPU profiling context manager for timing
class GPUProfiler:
    """
    A context manager for profiling GPU code execution time.
    This class is designed to measure the elapsed time of a code block,
    particularly when using GPU resources. It ensures synchronization
    of CUDA operations before and after the timed block to provide
    accurate timing results.
    Attributes:
        name (str): A descriptive name for the profiling block.
        start_time (float): The start time of the profiling block.
    Methods:
        __enter__():
            Starts the profiling timer. Synchronizes CUDA operations
            if a GPU is available.
        __exit__(*args):
            Stops the profiling timer, synchronizes CUDA operations
            if a GPU is available, and prints the elapsed time.
    Note:
        This profiler assumes that PyTorch is being used for GPU operations.
        Ensure that `torch` and `time` modules are imported before using this class.
    """

    def __init__(self, name):
        self.name = name
        self.start_time = None
        
    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.start_time = time.time()
        return self
        
    def __exit__(self, *args):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.time() - self.start_time
        print(f"[PROFILE] {self.name}: {elapsed:.3f}s")

def test():
    print('hhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh')