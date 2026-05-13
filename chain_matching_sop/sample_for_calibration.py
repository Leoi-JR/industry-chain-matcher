#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main script for similarity-based sampling for threshold calibration
Samples source texts across similarity bins for each chain
"""

import numpy as np
import pandas as pd
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from config import data_config
from utils.io_utils import load_json, save_json, load_npy_file


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description='Sample source texts across similarity bins for threshold calibration',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--start_similarity',
        type=float,
        default=data_config.START_SIMILARITY,
        help=f'Starting similarity value for binning (default: {data_config.START_SIMILARITY})'
    )

    parser.add_argument(
        '--bin_width',
        type=float,
        default=data_config.BIN_WIDTH,
        help=f'Width of each similarity bin (default: {data_config.BIN_WIDTH})'
    )

    parser.add_argument(
        '--samples_per_bin',
        type=int,
        default=data_config.SAMPLES_PER_BIN,
        help=f'Maximum number of samples per bin (default: {data_config.SAMPLES_PER_BIN})'
    )

    parser.add_argument(
        '--output',
        type=str,
        default=data_config.SAMPLING_OUTPUT_PATH,
        help='Output JSON file path (default: similarity_output/similarity_samples.json)'
    )

    parser.add_argument(
        '--no_progress',
        action='store_true',
        help='Hide progress bars'
    )

    return parser.parse_args()


def load_all_similarity_matrices(
    similarity_output_dir: Path,
    show_progress: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load and concatenate all similarity matrices from batches.
    
    Args:
        similarity_output_dir: Directory containing similarity output batches
        show_progress: Whether to show progress bar
    
    Returns:
        Tuple of (sim_spec_full, sim_other_full, sim_l0_full, global_ids)
    """
    print("=" * 70)
    print("STEP 1: Loading Similarity Matrices")
    print("=" * 70)
    
    # Load metadata to get number of batches
    metadata_path = similarity_output_dir / "similarity_metadata.json"
    metadata = load_json(metadata_path)
    num_batches = metadata['num_batches']
    m_spec = metadata['m_spec']
    m_other = metadata['m_other']
    
    print(f"Number of batches: {num_batches}")
    print(f"Number of spec chains: {m_spec}")
    print(f"Number of other chains: {m_other}")
    print()
    
    # Initialize lists to store batch data
    sim_spec_list = []
    sim_other_list = []
    sim_l0_list = []
    ids_list = []
    
    # Load all batches
    batch_dirs = sorted([d for d in similarity_output_dir.iterdir() if d.is_dir() and d.name.startswith('batch_')])
    
    iterator = tqdm(batch_dirs, desc="Loading batches") if show_progress else batch_dirs
    
    for batch_dir in iterator:
        # Load similarity matrices
        sim_spec = np.load(batch_dir / "sim_spec.npy")
        sim_other = np.load(batch_dir / "sim_other_parent.npy")
        sim_l0 = np.load(batch_dir / "sim_l0.npy")
        
        # Load batch metadata for IDs
        batch_metadata = load_json(batch_dir / "batch_metadata.json")
        ids = np.array(batch_metadata['ids'])
        
        # Append to lists
        sim_spec_list.append(sim_spec)
        sim_other_list.append(sim_other)
        sim_l0_list.append(sim_l0)
        ids_list.append(ids)
    
    # Concatenate along axis=1 (sources dimension)
    print("\nConcatenating matrices...")
    sim_spec_full = np.concatenate(sim_spec_list, axis=1)
    sim_other_full = np.concatenate(sim_other_list, axis=1)
    sim_l0_full = np.concatenate(sim_l0_list, axis=1)
    global_ids = np.concatenate(ids_list)
    
    print(f"  sim_spec shape: {sim_spec_full.shape}")
    print(f"  sim_other shape: {sim_other_full.shape}")
    print(f"  sim_l0 shape: {sim_l0_full.shape}")
    print(f"  Total sources: {len(global_ids)}")
    print()
    
    return sim_spec_full, sim_other_full, sim_l0_full, global_ids


class SourceDataMapper:
    """
    Memory-efficient mapper for source texts.
    Creates an ID index and caches file data for better performance.
    """

    def __init__(self, source_data_dir: Path, show_progress: bool = True):
        """
        Initialize mapper by creating ID-to-file mapping.

        Args:
            source_data_dir: Directory containing source data parquet files
            show_progress: Whether to show progress bar
        """
        self.source_data_dir = source_data_dir
        self.id_to_file = {}
        self.file_cache = {}  # Cache loaded files to avoid repeated I/O

        print("=" * 70)
        print("STEP 2: Creating Source Data ID Index")
        print("=" * 70)

        parquet_files = sorted(source_data_dir.glob(data_config.SOURCE_DATA_FILE_PATTERN))
        print(f"Found {len(parquet_files)} parquet files")
        print("Building ID index (memory-efficient)...")
        print()
        
        iterator = tqdm(parquet_files, desc="Indexing files") if show_progress else parquet_files
        
        for parquet_file in iterator:
            # Only load ID column to build index
            df = pd.read_parquet(parquet_file, columns=['id'])
            # df['id'] = df['id'].astype('int64')
            ids = df['id'].values
            
            # Map each ID to its source file
            for id_val in ids:
                self.id_to_file[id_val] = parquet_file
        
        print(f"  Total indexed IDs: {len(self.id_to_file)}")
        print()
    
    def get_texts(self, ids: np.ndarray) -> List[Tuple[int, str]]:
        """
        Get source texts for given IDs.

        Args:
            ids: Array of IDs to look up

        Returns:
            List of (id, source_text) tuples
        """
        # Group IDs by file to minimize file reads
        file_to_ids = {}
        for id_val in ids:
            if id_val in self.id_to_file:
                file_path = self.id_to_file[id_val]
                if file_path not in file_to_ids:
                    file_to_ids[file_path] = []
                file_to_ids[file_path].append(id_val)

        # Load each file once and extract needed texts
        id_to_text = {}
        for file_path, file_ids in file_to_ids.items():
            # Check cache first
            if file_path not in self.file_cache:
                # Load and cache the file
                df = pd.read_parquet(file_path, columns=['id', data_config.SOURCE_TEXT_FIELD])
                # df['id'] = df['id'].astype('int64')
                # Convert to dictionary for O(1) lookup
                self.file_cache[file_path] = dict(zip(df['id'], df[data_config.SOURCE_TEXT_FIELD]))

            # Get texts from cache
            file_dict = self.file_cache[file_path]
            for id_val in file_ids:
                if id_val in file_dict:
                    id_to_text[id_val] = file_dict[id_val]

        # Return (id, text) tuples in original order
        result = []
        for id_val in ids:
            if id_val in id_to_text:
                result.append((int(id_val), id_to_text[id_val]))

        return result


def create_similarity_bins(
    start_similarity: float,
    bin_width: float,
    max_similarity: float
) -> List[Tuple[float, float]]:
    """
    Create similarity bins from start to max.
    
    Args:
        start_similarity: Starting similarity value
        bin_width: Width of each bin
        max_similarity: Maximum similarity value
    
    Returns:
        List of (bin_start, bin_end) tuples
    """
    bins = []
    current = start_similarity
    
    while current < max_similarity:
        bin_end = min(current + bin_width, max_similarity + 0.001)  # Add small epsilon to include max
        bins.append((current, bin_end))
        current += bin_width

    return bins


def sample_from_bin(
    similarities: np.ndarray,
    global_ids: np.ndarray,
    bin_start: float,
    bin_end: float,
    samples_per_bin: int,
    mapper: SourceDataMapper,
    rng: np.random.Generator
) -> List[Tuple[int, str]]:
    """
    Sample source texts from a similarity bin.

    Args:
        similarities: Similarity scores for one chain
        global_ids: Global IDs corresponding to similarities
        bin_start: Start of bin (inclusive)
        bin_end: End of bin (exclusive)
        samples_per_bin: Maximum number of samples
        mapper: SourceDataMapper instance for looking up texts
        rng: Random number generator

    Returns:
        List of (id, source_text) tuples
    """
    # Find indices where similarity is in bin
    mask = (similarities >= bin_start) & (similarities < bin_end)
    bin_ids = global_ids[mask]
    
    # Sample up to samples_per_bin
    n_samples = min(len(bin_ids), samples_per_bin)
    if n_samples == 0:
        return []
    
    # Random sampling
    if n_samples < len(bin_ids):
        sampled_indices = rng.choice(len(bin_ids), size=n_samples, replace=False)
        sampled_ids = bin_ids[sampled_indices]
    else:
        sampled_ids = bin_ids

    # Look up texts using mapper (returns list of (id, text) tuples)
    samples = mapper.get_texts(sampled_ids)
    
    return samples


def sample_all_chains(
    sim_spec_full: np.ndarray,
    sim_other_full: np.ndarray,
    sim_l0_full: np.ndarray,
    global_ids: np.ndarray,
    spec_names: List[str],
    other_names: List[str],
    l0_name: str,
    mapper: SourceDataMapper,
    start_similarity: float,
    bin_width: float,
    samples_per_bin: int,
    show_progress: bool = True
) -> Dict[str, Dict[str, List[Tuple[int, str]]]]:
    """
    Sample source texts for all chains across similarity bins.

    Args:
        sim_spec_full: Full similarity matrix for spec chains (m_spec, n_total)
        sim_other_full: Full similarity matrix for other chains (m_other, n_total)
        sim_l0_full: Full similarity matrix for L0 (1, n_total)
        global_ids: Global IDs for all sources
        spec_names: List of spec chain names
        other_names: List of other chain names
        l0_name: L0 chain name
        mapper: SourceDataMapper instance for looking up texts
        start_similarity: Starting similarity value
        bin_width: Width of each similarity bin
        samples_per_bin: Maximum samples per bin
        show_progress: Whether to show progress bar

    Returns:
        Dictionary mapping chain names to bin samples.
        Each bin contains a list of (id, source_text) tuples.
    """
    print("=" * 70)
    print("STEP 3: Sampling Across Similarity Bins")
    print("=" * 70)
    
    result = {}
    rng = np.random.default_rng(42)  # Fixed seed for reproducibility
    
    # Process spec chains
    print(f"Processing {len(spec_names)} spec chains...")
    iterator = enumerate(spec_names)
    if show_progress:
        iterator = tqdm(list(iterator), desc="Spec chains")
    
    for idx, chain_name in iterator:
        similarities = sim_spec_full[idx, :]
        max_sim = similarities.max()
        
        # Create bins
        bins = create_similarity_bins(start_similarity, bin_width, max_sim)
        
        # Sample from each bin
        chain_samples = {}
        for bin_start, bin_end in bins:
            bin_key = f"[{bin_start:.3f}, {bin_end:.3f})"
            samples = sample_from_bin(
                similarities, global_ids, bin_start, bin_end,
                samples_per_bin, mapper, rng
            )
            if samples:  # Only add non-empty bins
                chain_samples[bin_key] = samples
        
        if chain_samples:  # Only add chains with samples
            result[chain_name] = chain_samples
    
    # Process other chains
    print(f"\nProcessing {len(other_names)} other chains...")
    iterator = enumerate(other_names)
    if show_progress:
        iterator = tqdm(list(iterator), desc="Other chains")
    
    for idx, chain_name in iterator:
        similarities = sim_other_full[idx, :]
        max_sim = similarities.max()
        
        # Create bins
        bins = create_similarity_bins(start_similarity, bin_width, max_sim)
        
        # Sample from each bin
        chain_samples = {}
        for bin_start, bin_end in bins:
            bin_key = f"[{bin_start:.3f}, {bin_end:.3f})"
            samples = sample_from_bin(
                similarities, global_ids, bin_start, bin_end,
                samples_per_bin, mapper, rng
            )
            if samples:  # Only add non-empty bins
                chain_samples[bin_key] = samples
        
        if chain_samples:  # Only add chains with samples
            result[chain_name] = chain_samples
    
    # Process L0
    print(f"\nProcessing L0: {l0_name}...")
    similarities = sim_l0_full[0, :]
    max_sim = similarities.max()
    
    # Create bins
    bins = create_similarity_bins(start_similarity, bin_width, max_sim)
    
    # Sample from each bin
    chain_samples = {}
    for bin_start, bin_end in bins:
        bin_key = f"[{bin_start:.3f}, {bin_end:.3f})"
        texts = sample_from_bin(
            similarities, global_ids, bin_start, bin_end,
            samples_per_bin, mapper, rng
        )
        if texts:  # Only add non-empty bins
            chain_samples[bin_key] = texts
    
    if chain_samples:  # Only add if has samples
        result[l0_name] = chain_samples
    
    print(f"\nTotal chains with samples: {len(result)}")
    print()
    
    return result


def main():
    """Main sampling function"""
    args = parse_args()
    
    print("=" * 70)
    print("CHAIN MATCHING SOP - SIMILARITY SAMPLING")
    print("=" * 70)
    print()
    print("Parameters:")
    print(f"  Start similarity: {args.start_similarity}")
    print(f"  Bin width: {args.bin_width}")
    print(f"  Samples per bin: {args.samples_per_bin}")
    print()
    
    show_progress = not args.no_progress

    # Define paths
    similarity_output_dir = data_config.SIMILARITY_OUTPUT_DIR
    source_data_dir = data_config.SOURCE_DATA_DIR
    metadata_path = data_config.METADATA_PATH

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = data_config.LLM_SIMILARITY_SAMPLES_PATH

    # Load metadata
    print("Loading metadata...")
    metadata = load_json(metadata_path)
    spec_names = metadata['spec_names']
    other_names = metadata['other_names']
    l0_name = metadata['l0_name']
    print(f"  Spec chains: {len(spec_names)}")
    print(f"  Other chains: {len(other_names)}")
    print(f"  L0: {l0_name}")
    print()

    # Step 1: Load similarity matrices
    sim_spec_full, sim_other_full, sim_l0_full, global_ids = load_all_similarity_matrices(
        similarity_output_dir, show_progress
    )

    # Step 2: Create source data mapper
    mapper = SourceDataMapper(source_data_dir, show_progress)
    
    # Step 3: Sample across similarity bins
    samples = sample_all_chains(
        sim_spec_full, sim_other_full, sim_l0_full, global_ids,
        spec_names, other_names, l0_name,
        mapper,
        args.start_similarity, args.bin_width, args.samples_per_bin,
        show_progress
    )
    
    # Step 4: Save output
    print("=" * 70)
    print("STEP 4: Saving Results")
    print("=" * 70)
    print(f"Output path: {output_path}")
    
    save_json(output_path, samples, indent=2)
    
    print(f"  File saved successfully")
    
    # Print summary statistics
    print()
    print("=" * 70)
    print("SAMPLING COMPLETE")
    print("=" * 70)
    print()
    print("Summary:")
    print(f"  Total chains sampled: {len(samples)}")
    
    total_bins = sum(len(bins) for bins in samples.values())
    total_samples = sum(
        len(texts) 
        for chain_bins in samples.values() 
        for texts in chain_bins.values()
    )
    
    print(f"  Total bins: {total_bins}")
    print(f"  Total samples: {total_samples}")
    print()
    print(f"Output saved to: {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()

