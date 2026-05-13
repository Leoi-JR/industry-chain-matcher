#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare indices for specific and "other" chains
"""

import numpy as np
from typing import List, Tuple
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.data_config import CHAIN_SEPARATOR, OTHER_IDENTIFIER


def prepare_indices(
    chain_names: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """
    Prepare indices for specific and "other" chains.
    
    Args:
        chain_names: Array of all chain names (m_total,)
    
    Returns:
        Tuple of:
        - spec_indices: Indices of specific chains in original array
        - other_indices: Indices of "other" chains in original array
        - spec_names: List of specific chain names
        - other_names: List of "other" chain names
    """
    spec_indices_list = []
    other_indices_list = []
    spec_names = []
    other_names = []
    
    for idx, chain_name in enumerate(chain_names):
        # Split chain name by separator
        parts = chain_name.split(CHAIN_SEPARATOR)
        
        # Check if last segment is "其他"
        if parts[-1] == OTHER_IDENTIFIER:
            other_indices_list.append(idx)
            other_names.append(chain_name)
        else:
            spec_indices_list.append(idx)
            spec_names.append(chain_name)
    
    # Convert to numpy arrays
    spec_indices = np.array(spec_indices_list, dtype=np.int64)
    other_indices = np.array(other_indices_list, dtype=np.int64)
    
    print(f"Found {len(spec_indices)} specific chains")
    print(f"Found {len(other_indices)} 'other' chains")
    print(f"Total: {len(spec_indices) + len(other_indices)} chains")
    
    return spec_indices, other_indices, spec_names, other_names


def get_parent_chain_name(chain_name: str) -> str:
    """
    Get parent chain name by removing the last segment.
    
    Args:
        chain_name: Full chain name (e.g., "A——B——C——其他")
    
    Returns:
        Parent chain name (e.g., "A——B——C")
    """
    parts = chain_name.split(CHAIN_SEPARATOR)
    
    if len(parts) <= 1:
        raise ValueError(f"Chain '{chain_name}' has no parent")
    
    # Remove last segment
    parent_parts = parts[:-1]
    parent_name = CHAIN_SEPARATOR.join(parent_parts)
    
    return parent_name


if __name__ == "__main__":
    # Test
    test_chains = np.array([
        "ExampleChain——A——B——C",
        "ExampleChain——A——B——其他",
        "ExampleChain——A——D",
        "ExampleChain——A——其他"
    ])
    
    spec_idx, other_idx, spec_names, other_names = prepare_indices(test_chains)
    
    print("\nSpecific chains:")
    for idx, name in zip(spec_idx, spec_names):
        print(f"  [{idx}] {name}")
    
    print("\n'Other' chains:")
    for idx, name in zip(other_idx, other_names):
        parent = get_parent_chain_name(name)
        print(f"  [{idx}] {name} -> parent: {parent}")

