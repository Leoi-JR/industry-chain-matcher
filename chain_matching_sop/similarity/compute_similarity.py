#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute similarity scores in batches
"""

import numpy as np
from pathlib import Path
from typing import List
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import data_config, sop_config
from utils.io_utils import (
    load_npz_file, save_npy_file, save_json, ensure_dir, load_json
)
from utils.matrix_ops import compute_cosine_similarity, is_gpu_available, get_gpu_info


def compute_batch_similarities(
    l0_embed: np.ndarray,
    spec_embeds: np.ndarray,
    other_parent_embeds: np.ndarray,
    doc_embeds: np.ndarray,
    normalize: bool = True,
    use_gpu: bool = False
) -> tuple:
    """
    Compute similarity scores for one batch.
    
    Args:
        l0_embed: L0 embedding (1, d)
        spec_embeds: Specific chain embeddings (m_spec, d)
        other_parent_embeds: Other chain parent embeddings (m_other, d)
        doc_embeds: Document embeddings for this batch (n_batch, d)
        normalize: Whether to normalize embeddings
        use_gpu: Whether to use GPU acceleration
    
    Returns:
        Tuple of (sim_l0, sim_spec, sim_other_parent)
    """
    # Compute L0 similarity: (1, d) @ (d, n_batch) -> (1, n_batch)
    sim_l0 = compute_cosine_similarity(l0_embed, doc_embeds, normalize=normalize, use_gpu=use_gpu)
    
    # Compute specific chain similarity: (m_spec, d) @ (d, n_batch) -> (m_spec, n_batch)
    sim_spec = compute_cosine_similarity(spec_embeds, doc_embeds, normalize=normalize, use_gpu=use_gpu)
    
    # Compute other parent similarity: (m_other, d) @ (d, n_batch) -> (m_other, n_batch)
    if other_parent_embeds.shape[0] > 0:
        sim_other_parent = compute_cosine_similarity(
            other_parent_embeds, doc_embeds, normalize=normalize, use_gpu=use_gpu
        )
    else:
        # No "other" chains
        sim_other_parent = np.zeros((0, doc_embeds.shape[0]), dtype=np.float32)
    
    return sim_l0, sim_spec, sim_other_parent


def save_batch_results(
    batch_idx: int,
    batch_ids: np.ndarray,
    sim_l0: np.ndarray,
    sim_spec: np.ndarray,
    sim_other_parent: np.ndarray,
    output_dir: Path
) -> None:
    """
    Save similarity results for one batch.
    
    Args:
        batch_idx: Batch index
        batch_ids: Source IDs for this batch
        sim_l0: L0 similarity matrix
        sim_spec: Specific chain similarity matrix
        sim_other_parent: Other chain parent similarity matrix
        output_dir: Output directory
    """
    # Create batch directory
    batch_dir = output_dir / f"batch_{batch_idx}"
    ensure_dir(batch_dir)
    
    # Save similarity matrices
    save_npy_file(batch_dir / "sim_l0.npy", sim_l0)
    save_npy_file(batch_dir / "sim_spec.npy", sim_spec)
    save_npy_file(batch_dir / "sim_other_parent.npy", sim_other_parent)
    
    # Save batch metadata
    metadata = {
        'batch_idx': int(batch_idx),
        'n_batch': len(batch_ids),
        'ids': batch_ids.tolist() if hasattr(batch_ids, 'tolist') else list(batch_ids),
        'sim_l0_shape': list(sim_l0.shape),
        'sim_spec_shape': list(sim_spec.shape),
        'sim_other_parent_shape': list(sim_other_parent.shape)
    }
    save_json(batch_dir / "batch_metadata.json", metadata)


def compute_all_similarities(
    show_progress: bool = True,
    use_gpu: bool = None
) -> None:
    """
    Compute all similarities for all batches.
    
    Args:
        show_progress: Whether to show progress bar
        use_gpu: Whether to use GPU (None = use config default)
    """
    print("=" * 70)
    print("SIMILARITY COMPUTATION")
    print("=" * 70)
    print()
    
    # ========================================================================
    # Step 0: GPU Setup
    # ========================================================================
    if use_gpu is None:
        use_gpu = sop_config.USE_GPU
    
    # Check GPU availability
    gpu_available = is_gpu_available()
    
    if use_gpu and not gpu_available:
        print("Warning: GPU requested but not available. Falling back to CPU.")
        use_gpu = False
    
    if use_gpu:
        gpu_info = get_gpu_info()
        print("GPU Configuration:")
        print(f"  GPU Enabled: Yes")
        print(f"  Device ID: {gpu_info.get('device_id', 'N/A')}")
        print(f"  Device Name: {gpu_info.get('device_name', 'N/A')}")
        print(f"  Total Memory: {gpu_info.get('total_memory_gb', 0):.2f} GB")
        print(f"  Free Memory: {gpu_info.get('free_memory_gb', 0):.2f} GB")
    else:
        print("GPU Configuration:")
        print(f"  GPU Enabled: No (using CPU)")
    print()
    
    # ========================================================================
    # Step 1: Load prepared data
    # ========================================================================
    print("Step 1: Loading prepared data...")
    
    # Load metadata
    metadata = load_json(data_config.METADATA_PATH)
    m_spec = metadata['m_spec']
    m_other = metadata['m_other']
    d = metadata['d']
    
    print(f"  m_spec: {m_spec}")
    print(f"  m_other: {m_other}")
    print(f"  d: {d}")
    
    # Load embeddings
    print("  Loading L0 embedding...")
    l0_data = load_npz_file(data_config.L0_EMBED_PATH)
    l0_embed = l0_data['embedding']
    
    print("  Loading specific chain embeddings...")
    spec_data = load_npz_file(data_config.SPEC_EMBEDS_PATH)
    spec_embeds = spec_data['embeddings']
    
    print("  Loading other chain parent embeddings...")
    other_data = load_npz_file(data_config.OTHER_PARENT_EMBEDS_PATH)
    other_parent_embeds = other_data['embeddings']
    
    print()
    
    # ========================================================================
    # Step 2: 查找信源嵌入向量文件
    # ========================================================================
    print("Step 2: Finding source embedding files...")

    source_files = sorted(
        data_config.SOURCE_EMBEDDINGS_DIR.glob(data_config.SOURCE_EMBEDDINGS_PATTERN)
    )

    print(f"  Found {len(source_files)} source embedding files")
    print()
    
    # ========================================================================
    # Step 3: Process each batch
    # ========================================================================
    print("Step 3: Computing similarities for each batch...")
    
    # Ensure output directory exists
    ensure_dir(data_config.SIMILARITY_OUTPUT_DIR)
    
    # 进度跟踪
    if show_progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(enumerate(source_files), total=len(source_files), desc="Processing batches")
        except ImportError:
            print("  (tqdm not available, showing simple progress)")
            iterator = enumerate(source_files)
            show_progress = False
    else:
        iterator = enumerate(source_files)

    total_sources = 0

    for batch_idx, source_file in iterator:
        # 加载信源嵌入向量
        source_data = load_npz_file(source_file)
        doc_embeds = source_data['embeddings']
        doc_ids = source_data['ids']

        n_batch = len(doc_ids)
        total_sources += n_batch

        if not show_progress:
            print(f"  Batch {batch_idx}: {source_file.name} ({n_batch} sources)")
        
        # Compute similarities
        sim_l0, sim_spec, sim_other_parent = compute_batch_similarities(
            l0_embed,
            spec_embeds,
            other_parent_embeds,
            doc_embeds,
            normalize=sop_config.NORMALIZE_EMBEDDINGS,
            use_gpu=use_gpu
        )
        
        # Save results
        save_batch_results(
            batch_idx,
            doc_ids,
            sim_l0,
            sim_spec,
            sim_other_parent,
            data_config.SIMILARITY_OUTPUT_DIR
        )
    
    print()
    
    # ========================================================================
    # Step 4: 保存全局元数据
    # ========================================================================
    print("Step 4: Saving global metadata...")

    sim_metadata = {
        'num_batches': len(source_files),
        'total_sources': total_sources,
        'm_spec': m_spec,
        'm_other': m_other,
        'd': d,
        'normalize_embeddings': sop_config.NORMALIZE_EMBEDDINGS,
        'gpu_used': use_gpu
    }
    
    save_json(data_config.SIMILARITY_OUTPUT_DIR / "similarity_metadata.json", sim_metadata)
    
    print()
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("=" * 70)
    print("SIMILARITY COMPUTATION COMPLETE")
    print("=" * 70)
    print()
    print("Summary:")
    print(f"  Processed {len(source_files)} batches")
    print(f"  Total sources: {total_sources}")
    print(f"  Similarity matrices saved to: {data_config.SIMILARITY_OUTPUT_DIR}")
    print()
    print("Next step:")
    print("  1. Determine thresholds by sampling and LLM review")
    print("  2. Run apply_matching.py to apply matching logic")
    print("=" * 70)


if __name__ == "__main__":
    compute_all_similarities()

