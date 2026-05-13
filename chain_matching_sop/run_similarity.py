#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main script for similarity computation
"""

import sys
import os
import argparse
from pathlib import Path


def setup_cuda_device(device_id):
    """
    Set CUDA_VISIBLE_DEVICES environment variable before importing any GPU libraries.
    
    Args:
        device_id: GPU device ID (integer or None for auto-select)
    """
    if device_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(device_id)
        print(f"Set CUDA_VISIBLE_DEVICES={device_id}")


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description='Compute similarity scores between chains and sources',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default settings (GPU auto-select if available)
  python run_similarity.py

  # Use specific GPU device
  python run_similarity.py --gpu_device 0

  # Force CPU computation
  python run_similarity.py --no_gpu

  # Hide progress bar
  python run_similarity.py --no_progress
        """
    )
    
    parser.add_argument(
        '--gpu_device',
        type=int,
        default=None,
        help='GPU device ID to use (default: None for auto-select)'
    )
    
    parser.add_argument(
        '--no_gpu',
        action='store_true',
        help='Force CPU computation (disable GPU)'
    )
    
    parser.add_argument(
        '--no_progress',
        action='store_true',
        help='Hide progress bar'
    )
    
    return parser.parse_args()


# Parse arguments and setup CUDA device BEFORE importing other modules
args = parse_args()
setup_cuda_device(args.gpu_device)

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from similarity.compute_similarity import compute_all_similarities


def main():
    """Main entry point for similarity computation"""
    # Determine whether to use GPU
    use_gpu = not args.no_gpu
    show_progress = not args.no_progress
    
    compute_all_similarities(
        show_progress=show_progress,
        use_gpu=use_gpu
    )


if __name__ == "__main__":
    main()

