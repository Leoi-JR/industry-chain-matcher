#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Apply thresholds and matching logic
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import data_config, sop_config
from utils.io_utils import (
    load_npz_file, load_npy_file, load_json, load_csv,
    save_npz_file, save_csv, ensure_dir
)
from utils.matrix_ops import (
    apply_threshold, compute_cascade_mask, compute_exclusion_mask,
    assemble_total_matrix
)


def load_thresholds(
    spec_names: List[str],
    other_names: List[str]
) -> Dict:
    """
    Load all threshold files.
    
    Args:
        spec_names: List of specific chain names
        other_names: List of "other" chain names
    
    Returns:
        Dictionary with threshold data
    """
    print("Loading thresholds...")
    
    # Load T_L0
    t_l0_data = load_json(data_config.THRESHOLD_L0_PATH)
    t_l0 = float(t_l0_data['T_L0'])
    print(f"  T_L0: {t_l0}")
    
    # Load Threshold_Spec
    df_spec = load_csv(data_config.THRESHOLD_SPEC_PATH)
    spec_threshold_map = dict(zip(df_spec['chain_name'], df_spec['threshold']))
    
    # Build threshold vector for specific chains (m_spec, 1)
    threshold_spec = np.zeros((len(spec_names), 1), dtype=np.float32)
    for idx, name in enumerate(spec_names):
        if name not in spec_threshold_map:
            raise ValueError(f"Threshold for specific chain '{name}' not found")
        threshold_spec[idx, 0] = float(spec_threshold_map[name])
    
    print(f"  Loaded {len(spec_names)} specific chain thresholds")
    
    # Load Threshold_Other
    df_other = load_csv(data_config.THRESHOLD_OTHER_PATH)
    other_threshold_map = dict(zip(df_other['chain_name'], df_other['threshold']))
    
    # Build threshold vector for "other" chains (m_other, 1)
    threshold_other = np.zeros((len(other_names), 1), dtype=np.float32)
    for idx, name in enumerate(other_names):
        if name not in other_threshold_map:
            raise ValueError(f"Threshold for 'other' chain '{name}' not found")
        threshold_other[idx, 0] = float(other_threshold_map[name])
    
    print(f"  Loaded {len(other_names)} 'other' chain thresholds")
    
    return {
        't_l0': t_l0,
        'threshold_spec': threshold_spec,
        'threshold_other': threshold_other
    }


def process_batch(
    batch_idx: int,
    sim_l0: np.ndarray,
    sim_spec: np.ndarray,
    sim_other_parent: np.ndarray,
    t_l0: float,
    threshold_spec: np.ndarray,
    threshold_other: np.ndarray,
    chain_type_vector: np.ndarray,
    spec_indices: np.ndarray,
    other_indices: np.ndarray,
    exclusion_matrix: np.ndarray,
    m_total: int
) -> tuple:
    """
    Process one batch to compute final matching results.
    
    Args:
        batch_idx: Batch index
        sim_l0: L0 similarity (1, n_batch)
        sim_spec: Specific chain similarity (m_spec, n_batch)
        sim_other_parent: Other parent similarity (m_other, n_batch)
        t_l0: L0 threshold
        threshold_spec: Specific chain thresholds (m_spec, 1)
        threshold_other: Other chain thresholds (m_other, 1)
        chain_type_vector: Chain type vector (m_total, 1)
        spec_indices: Specific chain indices
        other_indices: Other chain indices
        exclusion_matrix: Exclusion mapping matrix (m_other, m_spec)
        m_total: Total number of chains
    
    Returns:
        Tuple of (result_matrix, final_mask_total, final_sim_total)
    """
    n_batch = sim_l0.shape[1]
    m_spec = sim_spec.shape[0]
    m_other = sim_other_parent.shape[0]
    
    # ========================================================================
    # Stage 2: Generate Masks (Original Judgments)
    # ========================================================================
    
    # Mask_L0 (1, n_batch)
    mask_l0 = sim_l0 > t_l0
    
    # Mask_Spec (m_spec, n_batch)
    mask_spec = apply_threshold(sim_spec, threshold_spec)
    
    # Mask_Other_Parent (m_other, n_batch)
    if m_other > 0:
        mask_other_parent = apply_threshold(sim_other_parent, threshold_other)
    else:
        mask_other_parent = np.zeros((0, n_batch), dtype=bool)
    
    # ========================================================================
    # Stage 3: L0 Cascade Filtering (A/B Group Logic)
    # ========================================================================
    
    # Extract type vectors for spec and other chains
    type_spec = chain_type_vector[spec_indices]  # (m_spec, 1)
    type_other = chain_type_vector[other_indices] if m_other > 0 else np.zeros((0, 1))
    
    # Broadcast L0 mask
    mask_l0_spec = np.repeat(mask_l0, m_spec, axis=0)  # (m_spec, n_batch)
    mask_l0_other = np.repeat(mask_l0, m_other, axis=0) if m_other > 0 else np.zeros((0, n_batch), dtype=bool)
    
    # 计算级联掩码
    cascade_mask_spec = compute_cascade_mask(mask_l0_spec, type_spec)  # (m_spec, n_batch)
    cascade_mask_other = compute_cascade_mask(mask_l0_other, type_other) if m_other > 0 else np.zeros((0, n_batch), dtype=bool)
    
    # ========================================================================
    # Stage 4: "Other" Exclusion Logic
    # ========================================================================
    
    # Compute No_Sibling_Match (m_other, n_batch)
    if m_other > 0:
        no_sibling_match = compute_exclusion_mask(mask_spec, exclusion_matrix)
    else:
        no_sibling_match = np.zeros((0, n_batch), dtype=bool)
    
    # ========================================================================
    # Stage 5: Assembly and Final Calculation
    # ========================================================================
    
    # Final_Mask_Spec (m_spec, n_batch)
    final_mask_spec = mask_spec & cascade_mask_spec
    
    # Final_Mask_Other (m_other, n_batch)
    if m_other > 0:
        final_mask_other = mask_other_parent & cascade_mask_other & no_sibling_match
    else:
        final_mask_other = np.zeros((0, n_batch), dtype=bool)
    
    # Assemble Final_Mask_Total (m_total, n_batch)
    final_mask_total = assemble_total_matrix(spec_indices, final_mask_spec, m_total, n_batch)
    if m_other > 0:
        final_mask_total[other_indices, :] = final_mask_other
    
    # Assemble Final_Sim_Total (m_total, n_batch)
    final_sim_total = assemble_total_matrix(spec_indices, sim_spec, m_total, n_batch)
    if m_other > 0:
        final_sim_total[other_indices, :] = sim_other_parent
    
    # Result_Matrix = Final_Mask_Total * Final_Sim_Total
    result_matrix = final_mask_total.astype(float) * final_sim_total
    
    return result_matrix, final_mask_total, final_sim_total


def apply_matching_logic(show_progress: bool = True) -> None:
    """
    Apply matching logic to all batches.
    
    Args:
        show_progress: Whether to show progress bar
    """
    print("=" * 70)
    print("MATCHING LOGIC APPLICATION")
    print("=" * 70)
    print()
    
    # ========================================================================
    # Step 1: Load prepared data
    # ========================================================================
    print("Step 1: Loading prepared data...")
    
    # Load metadata
    metadata = load_json(data_config.METADATA_PATH)
    m_total = metadata['m_total']
    m_spec = metadata['m_spec']
    m_other = metadata['m_other']
    spec_names = metadata['spec_names']
    other_names = metadata['other_names']
    
    print(f"  m_total: {m_total}")
    print(f"  m_spec: {m_spec}")
    print(f"  m_other: {m_other}")
    
    # Load indices
    spec_indices = load_npy_file(data_config.SPEC_INDICES_PATH)
    other_indices = load_npy_file(data_config.OTHER_INDICES_PATH)
    
    # Load chain type vector
    chain_type_vector = load_npy_file(data_config.CHAIN_TYPE_VECTOR_PATH)
    
    # Load exclusion matrix
    exclusion_data = load_npz_file(data_config.EXCLUSION_MAPPING_MATRIX_PATH)
    exclusion_matrix = exclusion_data['matrix']
    
    print()
    
    # ========================================================================
    # Step 2: Load thresholds
    # ========================================================================
    print("Step 2: Loading thresholds...")
    
    thresholds = load_thresholds(spec_names, other_names)
    t_l0 = thresholds['t_l0']
    threshold_spec = thresholds['threshold_spec']
    threshold_other = thresholds['threshold_other']
    
    print()
    
    # ========================================================================
    # Step 3: Load similarity metadata
    # ========================================================================
    print("Step 3: Loading similarity metadata...")

    sim_metadata = load_json(data_config.SIMILARITY_OUTPUT_DIR / "similarity_metadata.json")
    num_batches = sim_metadata['num_batches']
    total_sources = sim_metadata['total_sources']
    
    print(f"  Number of batches: {num_batches}")
    print(f"  Total sources: {total_sources}")
    print()
    
    # ========================================================================
    # Step 4: Process each batch
    # ========================================================================
    print("Step 4: Processing batches...")
    
    # Ensure output directory exists
    ensure_dir(data_config.MATCHING_OUTPUT_DIR)
    
    # Progress tracking
    if show_progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(range(num_batches), desc="Processing batches")
        except ImportError:
            print("  (tqdm not available, showing simple progress)")
            iterator = range(num_batches)
            show_progress = False
    else:
        iterator = range(num_batches)
    
    # Statistics tracking
    total_matches = 0
    batch_summaries = []
    
    for batch_idx in iterator:
        batch_dir = data_config.SIMILARITY_OUTPUT_DIR / f"batch_{batch_idx}"
        
        # Load similarity matrices
        sim_l0 = load_npy_file(batch_dir / "sim_l0.npy")
        sim_spec = load_npy_file(batch_dir / "sim_spec.npy")
        sim_other_parent = load_npy_file(batch_dir / "sim_other_parent.npy")
        
        # Load batch metadata
        batch_metadata = load_json(batch_dir / "batch_metadata.json")
        batch_ids = np.array(batch_metadata['ids'])
        
        if not show_progress:
            print(f"  Batch {batch_idx}: {len(batch_ids)} sources")
        
        # Process batch
        result_matrix, final_mask_total, final_sim_total = process_batch(
            batch_idx,
            sim_l0,
            sim_spec,
            sim_other_parent,
            t_l0,
            threshold_spec,
            threshold_other,
            chain_type_vector,
            spec_indices,
            other_indices,
            exclusion_matrix,
            m_total
        )
        
        # Save batch results
        output_path = data_config.MATCHING_OUTPUT_DIR / f"result_matrix_batch_{batch_idx}.npz"
        save_npz_file(
            output_path,
            {
                'result_matrix': result_matrix,
                'final_mask': final_mask_total,
                'final_sim': final_sim_total,
                'source_ids': batch_ids
            },
            compressed=True
        )
        
        # Track statistics
        num_matches = np.sum(result_matrix > 0)
        total_matches += num_matches
        
        batch_summaries.append({
            'batch_idx': batch_idx,
            'n_sources': len(batch_ids),
            'num_matches': int(num_matches),
            'avg_matches_per_source': float(num_matches / len(batch_ids)) if len(batch_ids) > 0 else 0
        })
    
    print()
    
    # ========================================================================
    # Step 5: Save summary
    # ========================================================================
    print("Step 5: Saving summary...")
    
    summary_df = pd.DataFrame(batch_summaries)
    save_csv(data_config.MATCHING_SUMMARY_PATH, summary_df)
    
    print()
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("=" * 70)
    print("MATCHING COMPLETE")
    print("=" * 70)
    print()
    print("Summary:")
    print(f"  Processed {num_batches} batches")
    print(f"  Total sources: {total_sources}")
    print(f"  Total matches: {total_matches}")
    print(f"  Average matches per source: {total_matches / total_sources:.2f}")
    print()
    print(f"Results saved to: {data_config.MATCHING_OUTPUT_DIR}")
    print(f"Summary saved to: {data_config.MATCHING_SUMMARY_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    apply_matching_logic()

