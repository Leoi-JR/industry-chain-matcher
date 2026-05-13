#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main script for chain configuration preparation
Generates all required configuration matrices and indices
"""

import numpy as np
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from config import data_config
from utils.io_utils import (
    load_npz_file, save_npz_file, save_npy_file, save_json, ensure_dir, load_csv
)
from data_preparation.prepare_indices import prepare_indices
from data_preparation.prepare_embeddings import prepare_embeddings
from data_preparation.prepare_exclusion import prepare_exclusion_matrix


def prepare_chain_type_vector(
    chain_names: np.ndarray,
    chain_type_df
) -> np.ndarray:
    """
    Prepare chain type vector (m_total, 1).
    
    Args:
        chain_names: All chain names (m_total,)
        chain_type_df: DataFrame with chain_name and type columns
    
    Returns:
        Chain type vector where 1=Type A, 0=Type B
    """
    print("Preparing chain type vector...")
    
    # Create mapping from chain name to type
    chain_type_map = dict(zip(chain_type_df['chain_name'], chain_type_df['type']))
    
    # Build vector
    chain_type_vector = np.zeros((len(chain_names), 1), dtype=np.float32)
    
    for idx, chain_name in enumerate(chain_names):
        if chain_name not in chain_type_map:
            raise ValueError(f"Chain '{chain_name}' not found in type classification")
        
        chain_type = chain_type_map[chain_name]
        if chain_type == data_config.CHAIN_TYPE_A:
            chain_type_vector[idx, 0] = 1.0
        elif chain_type == data_config.CHAIN_TYPE_B:
            chain_type_vector[idx, 0] = 0.0
        else:
            raise ValueError(f"Invalid chain type '{chain_type}' for chain '{chain_name}'")
    
    type_a_count = np.sum(chain_type_vector)
    type_b_count = len(chain_names) - type_a_count
    
    print(f"  Type A (requires L0 constraint): {int(type_a_count)}")
    print(f"  Type B (no L0 constraint): {int(type_b_count)}")
    
    return chain_type_vector


def main():
    """Main data preparation function"""
    print("=" * 70)
    print("CHAIN MATCHING SOP - DATA PREPARATION")
    print("=" * 70)
    print()
    
    # ========================================================================
    # Step 1: Load chain embeddings
    # ========================================================================
    print("Step 1: Loading chain embeddings...")
    print(f"  File: {data_config.CHAIN_EMBEDDINGS_PATH}")
    
    chain_data = load_npz_file(data_config.CHAIN_EMBEDDINGS_PATH)
    chain_names = chain_data['chain_names']
    chain_embeddings = chain_data['embeddings']
    
    m_total = len(chain_names)
    d = chain_embeddings.shape[1]
    
    print(f"  Loaded {m_total} chains with embedding dimension {d}")
    print()
    
    # ========================================================================
    # Step 2: Prepare indices
    # ========================================================================
    print("Step 2: Preparing indices...")
    
    spec_indices, other_indices, spec_names, other_names = prepare_indices(chain_names)
    
    m_spec = len(spec_indices)
    m_other = len(other_indices)
    
    print()
    
    # ========================================================================
    # Step 3: Prepare embeddings
    # ========================================================================
    print("Step 3: Preparing embeddings...")
    
    l0_embed, spec_embeds, other_embeds, l0_name = prepare_embeddings(
        chain_names, chain_embeddings, spec_indices, other_indices,
        data_config.L0_EMBEDDINGS_PATH
    )
    
    print()
    
    # ========================================================================
    # Step 4: Prepare exclusion matrix
    # ========================================================================
    print("Step 4: Preparing exclusion mapping matrix...")
    
    all_chain_names_list = list(chain_names)
    exclusion_matrix = prepare_exclusion_matrix(
        spec_names, other_names, all_chain_names_list
    )
    
    print()
    
    # ========================================================================
    # Step 5: Load and prepare chain type vector
    # ========================================================================
    print("Step 5: Preparing chain type vector...")
    print(f"  File: {data_config.CHAIN_TYPE_CLASSIFICATION_PATH}")
    
    chain_type_df = load_csv(data_config.CHAIN_TYPE_CLASSIFICATION_PATH)
    chain_type_vector = prepare_chain_type_vector(chain_names, chain_type_df)
    
    print()
    
    # ========================================================================
    # Step 6: Save all prepared data
    # ========================================================================
    print("Step 6: Saving prepared data...")
    
    # Ensure output directory exists
    ensure_dir(data_config.CONFIG_MATRICES_DIR)
    
    # Save L0 embedding
    print(f"  Saving L0 embedding to {data_config.L0_EMBED_PATH.name}")
    save_npz_file(
        data_config.L0_EMBED_PATH,
        {'embedding': l0_embed},
        compressed=True
    )
    
    # Save indices
    print(f"  Saving spec indices to {data_config.SPEC_INDICES_PATH.name}")
    save_npy_file(data_config.SPEC_INDICES_PATH, spec_indices)
    
    print(f"  Saving other indices to {data_config.OTHER_INDICES_PATH.name}")
    save_npy_file(data_config.OTHER_INDICES_PATH, other_indices)
    
    # Save embeddings
    print(f"  Saving spec embeddings to {data_config.SPEC_EMBEDS_PATH.name}")
    save_npz_file(
        data_config.SPEC_EMBEDS_PATH,
        {'embeddings': spec_embeds, 'chain_names': np.array(spec_names)},
        compressed=True
    )
    
    print(f"  Saving other embeddings to {data_config.OTHER_PARENT_EMBEDS_PATH.name}")
    save_npz_file(
        data_config.OTHER_PARENT_EMBEDS_PATH,
        {'embeddings': other_embeds, 'chain_names': np.array(other_names)},
        compressed=True
    )
    
    # Save exclusion matrix
    print(f"  Saving exclusion matrix to {data_config.EXCLUSION_MAPPING_MATRIX_PATH.name}")
    save_npz_file(
        data_config.EXCLUSION_MAPPING_MATRIX_PATH,
        {'matrix': exclusion_matrix},
        compressed=True
    )
    
    # Save chain type vector
    print(f"  Saving chain type vector to {data_config.CHAIN_TYPE_VECTOR_PATH.name}")
    save_npy_file(data_config.CHAIN_TYPE_VECTOR_PATH, chain_type_vector)
    
    # Save metadata
    metadata = {
        'm_total': int(m_total),
        'm_spec': int(m_spec),
        'm_other': int(m_other),
        'd': int(d),
        'chain_names': list(chain_names),
        'spec_names': spec_names,
        'other_names': other_names,
        'l0_name': l0_name
    }
    
    print(f"  Saving metadata to {data_config.METADATA_PATH.name}")
    save_json(data_config.METADATA_PATH, metadata)
    
    print()
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("=" * 70)
    print("DATA PREPARATION COMPLETE")
    print("=" * 70)
    print()
    print("Summary:")
    print(f"  Total chains (m_total): {m_total}")
    print(f"  Specific chains (m_spec): {m_spec}")
    print(f"  'Other' chains (m_other): {m_other}")
    print(f"  Embedding dimension (d): {d}")
    print()
    print(f"All prepared data saved to: {data_config.CONFIG_MATRICES_DIR}")
    print()
    print("Next step: Run run_similarity.py to compute similarity scores")
    print("=" * 70)


if __name__ == "__main__":
    main()

