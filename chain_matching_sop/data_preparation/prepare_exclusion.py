#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare Exclusion_Mapping_Matrix for "other" chain logic
"""

import numpy as np
from typing import List, Set
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.data_config import CHAIN_SEPARATOR
from data_preparation.prepare_indices import get_parent_chain_name


def get_chain_level(chain_name: str) -> int:
    """
    Get the level of a chain (number of segments).
    
    Args:
        chain_name: Chain name (e.g., "A——B——C" has level 3)
    
    Returns:
        Level number
    """
    parts = chain_name.split(CHAIN_SEPARATOR)
    return len(parts)


def get_siblings(
    chain_name: str,
    all_chain_names: List[str]
) -> List[str]:
    """
    Get sibling chains (chains with same parent).
    
    Args:
        chain_name: Target chain name
        all_chain_names: All available chain names
        same_level_only: If True, only return siblings at same level
    
    Returns:
        List of sibling chain names (excluding the input chain itself)
    """
    # Get parent
    parent_name = get_parent_chain_name(chain_name)
    parent_parts = parent_name.split(CHAIN_SEPARATOR)
    
    # Find all chains with the same parent
    siblings = []
    for candidate in all_chain_names:
        # Skip self
        if candidate == chain_name:
            continue
        
        # Check if candidate has same parent
        if candidate.startswith(parent_name + CHAIN_SEPARATOR):
            candidate_parts = candidate.split(CHAIN_SEPARATOR)
            
            # Direct child should have exactly one more segment than parent
            # (this automatically means same level as chain_name)
            if len(candidate_parts) == len(parent_parts) + 1:
                siblings.append(candidate)
    
    return siblings


def prepare_exclusion_matrix(
    spec_names: List[str],
    other_names: List[str],
    all_chain_names: List[str]
) -> np.ndarray:
    """
    Prepare Exclusion_Mapping_Matrix (m_other, m_spec).
    
    For each "other" chain, mark which specific chains are its siblings
    (same-level children of the same parent).
    
    Args:
        spec_names: List of specific chain names (m_spec,)
        other_names: List of "other" chain names (m_other,)
        all_chain_names: All chain names for sibling lookup
    
    Returns:
        Exclusion matrix of shape (m_other, m_spec) where
        E[i, j] = 1 if spec_chain[j] is a sibling to other_chain[i]
    """
    m_other = len(other_names)
    m_spec = len(spec_names)
    
    # Initialize matrix
    exclusion_matrix = np.zeros((m_other, m_spec), dtype=np.float32)
    
    # Create mapping from spec name to index
    spec_name_to_idx = {name: idx for idx, name in enumerate(spec_names)}
    
    print(f"Building exclusion matrix ({m_other} x {m_spec})...")
    
    # For each "other" chain
    for i, other_name in enumerate(other_names):
        # Get siblings (same-level specific chains with same parent)
        siblings = get_siblings(other_name, all_chain_names)
        
        # Mark siblings in the matrix
        sibling_count = 0
        for sibling_name in siblings:
            # Check if sibling is a specific chain
            if sibling_name in spec_name_to_idx:
                j = spec_name_to_idx[sibling_name]
                exclusion_matrix[i, j] = 1.0
                sibling_count += 1
        
        if sibling_count > 0:
            print(f"  '{other_name}' excludes {sibling_count} specific siblings")
    
    # Count statistics
    total_exclusions = np.sum(exclusion_matrix)
    print(f"\nTotal exclusion relationships: {int(total_exclusions)}")
    
    # Check for "other" chains with no siblings (warning)
    chains_without_siblings = np.sum(exclusion_matrix, axis=1) == 0
    if np.any(chains_without_siblings):
        count = np.sum(chains_without_siblings)
        print(f"Warning: {count} 'other' chains have no specific siblings to exclude")
    
    return exclusion_matrix


if __name__ == "__main__":
    # Test
    from prepare_indices import prepare_indices
    
    test_chains = [
        "ExampleChain",
        "ExampleChain——A",
        "ExampleChain——A——B",
        "ExampleChain——A——B——C",
        "ExampleChain——A——B——D",
        "ExampleChain——A——B——其他",  # Should exclude C and D
        "ExampleChain——A——E",
        "ExampleChain——A——F",
        "ExampleChain——A——其他",  # Should exclude B, E, F
    ]
    
    test_chains_array = np.array(test_chains)
    
    # Prepare indices
    spec_idx, other_idx, spec_names, other_names = prepare_indices(test_chains_array)
    
    print("\nSpecific chains:")
    for name in spec_names:
        print(f"  {name}")
    
    print("\n'Other' chains:")
    for name in other_names:
        print(f"  {name}")
    
    # Prepare exclusion matrix
    print("\n" + "="*70)
    exclusion_mat = prepare_exclusion_matrix(spec_names, other_names, test_chains)
    
    print("\nExclusion matrix:")
    print(f"Shape: {exclusion_mat.shape}")
    print(exclusion_mat)
    
    # Verify
    print("\n" + "="*70)
    print("Verification:")
    for i, other_name in enumerate(other_names):
        excluded_indices = np.where(exclusion_mat[i, :] == 1)[0]
        if len(excluded_indices) > 0:
            excluded_names = [spec_names[j] for j in excluded_indices]
            print(f"\n'{other_name}' excludes:")
            for name in excluded_names:
                print(f"  - {name}")

