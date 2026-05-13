#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare embeddings for L0, specific chains, and "other" chains
"""

import numpy as np
from typing import Tuple
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.io_utils import load_npz_file


def prepare_embeddings(
    chain_names: np.ndarray,
    chain_embeddings: np.ndarray,
    spec_indices: np.ndarray,
    other_indices: np.ndarray,
    l0_embedding_path: Path
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """
    Prepare embeddings for L0, specific chains, and "other" chains.
    
    Key understanding:
    - L0 comes from separate file (not in chain_embeddings)
    - "其他" chains in main file already use parent definitions
    - No parent lookup needed - use "其他" embeddings directly
    
    Args:
        chain_names: All chain names from main file (m_total,)
        chain_embeddings: All chain embeddings from main file (m_total, d)
        spec_indices: Indices of specific chains
        other_indices: Indices of "other" chains
        l0_embedding_path: Path to separate L0 embedding file
    
    Returns:
        Tuple of:
        - l0_embed: L0 embedding (1, d)
        - spec_embeds: Specific chain embeddings (m_spec, d)
        - other_embeds: "Other" chain embeddings (m_other, d) - already parent-based
        - l0_name: L0 chain name
    """
    # Load L0 from separate file
    print("  Loading L0 embedding from separate file...")
    l0_data = load_npz_file(l0_embedding_path)
    
    # Extract L0 name and embedding
    l0_names = l0_data['chain_names']
    if len(l0_names) != 1:
        raise ValueError(
            f"L0 file should contain exactly 1 chain, found {len(l0_names)}"
        )
    
    l0_name = l0_names[0]
    l0_embed = l0_data['embeddings'][0:1, :]  # Shape: (1, d)
    
    print(f"  L0 name: {l0_name}")
    print(f"  L0 embedding shape: {l0_embed.shape}")
    
    # Extract specific chain embeddings
    spec_embeds = chain_embeddings[spec_indices, :]
    print(f"  Specific chain embeddings shape: {spec_embeds.shape}")
    
    # Extract "other" chain embeddings
    # Key: These embeddings are already based on parent definitions
    # We use them directly, no parent lookup needed
    other_embeds = chain_embeddings[other_indices, :]
    print(f"  'Other' chain embeddings shape: {other_embeds.shape}")
    
    return l0_embed, spec_embeds, other_embeds, l0_name


if __name__ == "__main__":
    # Test
    from prepare_indices import prepare_indices
    
    # Simulate actual data structure
    test_chains = np.array([
        "ExampleChain——A——B——C",
        "ExampleChain——A——B——D",
        "ExampleChain——A——B——其他",  # Embedding already based on "ExampleChain——A——B"
        "ExampleChain——A——E",
        "ExampleChain——A——其他"  # Embedding already based on "ExampleChain——A"
    ])

    # Note: L0 "ExampleChain" is NOT in this list
    
    # Create dummy embeddings
    d = 8
    test_embeddings = np.random.randn(len(test_chains), d)
    
    # Create dummy L0 file
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
        temp_l0_path = f.name
        np.savez(temp_l0_path, 
                 chain_names=np.array(['ExampleChain']),
                 embeddings=np.random.randn(1, d))
    
    try:
        # Prepare indices
        spec_idx, other_idx, spec_names, other_names = prepare_indices(test_chains)
        
        print("\nTest data:")
        print(f"Total chains: {len(test_chains)}")
        print(f"Specific chains: {len(spec_idx)}")
        print(f"'Other' chains: {len(other_idx)}")
        
        # Prepare embeddings
        print("\nPreparing embeddings...")
        l0_emb, spec_embs, other_embs, l0_name = prepare_embeddings(
            test_chains, test_embeddings, spec_idx, other_idx, 
            Path(temp_l0_path)
        )
        
        print("\nResults:")
        print(f"L0 name: {l0_name}")
        print(f"L0 embedding shape: {l0_emb.shape}")
        print(f"Specific embeddings shape: {spec_embs.shape}")
        print(f"Other embeddings shape: {other_embs.shape}")
        
        # Verify
        assert l0_name == 'ExampleChain'
        assert l0_emb.shape == (1, d)
        assert spec_embs.shape == (len(spec_idx), d)
        assert other_embs.shape == (len(other_idx), d)
        
        print("\n✓ All tests passed!")
        
    finally:
        # Cleanup
        os.unlink(temp_l0_path)
