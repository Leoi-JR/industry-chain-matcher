#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validate input files and data formats
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import data_config
from utils.io_utils import load_npz_file, load_csv, load_json


def validate_l0_embeddings() -> Dict:
    """
    Validate L0 embeddings file.
    
    Returns:
        Dictionary with L0 data
    """
    print("Validating L0 embeddings...")
    
    if not data_config.L0_EMBEDDINGS_PATH.exists():
        raise FileNotFoundError(
            f"L0 embeddings file not found: {data_config.L0_EMBEDDINGS_PATH}"
        )
    
    # Load file
    data = load_npz_file(data_config.L0_EMBEDDINGS_PATH)
    
    # Check required keys
    required_keys = ['chain_names', 'embeddings']
    for key in required_keys:
        if key not in data:
            raise ValueError(f"Missing required key '{key}' in L0 embeddings file")
    
    l0_names = data['chain_names']
    l0_embeddings = data['embeddings']
    
    # Should have exactly 1 chain (the L0)
    if len(l0_names) != 1:
        raise ValueError(
            f"L0 file should contain exactly 1 chain, found {len(l0_names)}"
        )
    
    l0_name = l0_names[0]
    d_l0 = l0_embeddings.shape[1]
    
    print(f"  ✓ L0: {l0_name}")
    print(f"  ✓ Embedding dimension: {d_l0}")
    
    # Check for NaN values
    if np.any(np.isnan(l0_embeddings)):
        raise ValueError("L0 embedding contains NaN values")
    
    return {
        'l0_name': l0_name,
        'd': d_l0,
        'embedding': l0_embeddings
    }


def validate_chain_embeddings() -> Dict:
    """
    Validate chain embeddings file (detailed chains only, no L0).
    
    Returns:
        Dictionary with validation results and metadata
    """
    print("Validating chain embeddings...")
    
    if not data_config.CHAIN_EMBEDDINGS_PATH.exists():
        raise FileNotFoundError(
            f"Chain embeddings file not found: {data_config.CHAIN_EMBEDDINGS_PATH}"
        )
    
    # Load file
    data = load_npz_file(data_config.CHAIN_EMBEDDINGS_PATH)
    
    # Check required keys
    required_keys = ['chain_names', 'embeddings']
    for key in required_keys:
        if key not in data:
            raise ValueError(f"Missing required key '{key}' in chain embeddings file")
    
    chain_names = data['chain_names']
    embeddings = data['embeddings']
    
    # Validate shapes
    if chain_names.ndim != 1:
        raise ValueError(f"chain_names should be 1D, got shape {chain_names.shape}")
    
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings should be 2D, got shape {embeddings.shape}")
    
    m_total = len(chain_names)
    if embeddings.shape[0] != m_total:
        raise ValueError(
            f"Mismatch: {m_total} chain names but {embeddings.shape[0]} embeddings"
        )
    
    d = embeddings.shape[1]
    
    print(f"  ✓ Found {m_total} detailed chains with embedding dimension {d}")
    print(f"  ✓ Note: This file contains only detailed chains, not L0")
    
    # Check for NaN values
    if np.any(np.isnan(embeddings)):
        raise ValueError("Chain embeddings contain NaN values")
    
    return {
        'm_total': m_total,
        'd': d,
        'chain_names': chain_names,
        'embeddings': embeddings
    }


def validate_source_embeddings() -> Dict:
    """
    验证源文本嵌入向量目录和文件。

    支持多种源文本类型。

    Returns:
        Dictionary with validation results
    """
    print("Validating source embeddings...")

    if not data_config.SOURCE_EMBEDDINGS_DIR.exists():
        raise FileNotFoundError(
            f"Source embeddings directory not found: {data_config.SOURCE_EMBEDDINGS_DIR}"
        )

    # 查找所有源文本嵌入向量文件
    files = sorted(data_config.SOURCE_EMBEDDINGS_DIR.glob(data_config.SOURCE_EMBEDDINGS_PATTERN))

    if len(files) == 0:
        raise FileNotFoundError(
            f"No source embedding files found matching pattern: {data_config.SOURCE_EMBEDDINGS_PATTERN}"
        )

    print(f"  ✓ Found {len(files)} source embedding files")

    # 验证第一个文件作为样本
    first_file = files[0]
    data = load_npz_file(first_file)

    required_keys = ['ids', 'embeddings']
    for key in required_keys:
        if key not in data:
            raise ValueError(f"Missing required key '{key}' in source embeddings file")

    d = data['embeddings'].shape[1]
    print(f"  ✓ Embedding dimension: {d}")

    # 统计总源文本数量
    total_sources = 0
    for file in files:
        data = load_npz_file(file)
        total_sources += len(data['ids'])

    print(f"  ✓ Total sources across all files: {total_sources}")

    return {
        'num_files': len(files),
        'total_sources': total_sources,
        'd': d,
        'files': files
    }


def validate_chain_type_classification(chain_names: List[str]) -> pd.DataFrame:
    """
    Validate chain type classification file.
    
    Args:
        chain_names: List of all chain names for validation
    
    Returns:
        DataFrame with chain type data
    """
    print("Validating chain type classification...")
    
    if not data_config.CHAIN_TYPE_CLASSIFICATION_PATH.exists():
        raise FileNotFoundError(
            f"Chain type classification file not found: {data_config.CHAIN_TYPE_CLASSIFICATION_PATH}"
        )
    
    df = load_csv(data_config.CHAIN_TYPE_CLASSIFICATION_PATH)
    
    # Check required columns
    required_cols = ['chain_name', 'type']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in chain type file")
    
    # Check values
    valid_types = {data_config.CHAIN_TYPE_A, data_config.CHAIN_TYPE_B}
    invalid_types = set(df['type']) - valid_types
    if invalid_types:
        raise ValueError(
            f"Invalid chain types found: {invalid_types}. "
            f"Valid types are: {valid_types}"
        )
    
    # Check all chains are present
    chain_set = set(chain_names)
    type_chain_set = set(df['chain_name'])
    
    missing = chain_set - type_chain_set
    if missing:
        raise ValueError(
            f"{len(missing)} chains missing from type classification: {list(missing)[:5]}..."
        )
    
    extra = type_chain_set - chain_set
    if extra:
        print(f"  Warning: {len(extra)} extra chains in type file (will be ignored)")
    
    print(f"  ✓ Type classification covers all {len(chain_names)} chains")
    
    type_counts = df['type'].value_counts()
    print(f"  ✓ Type A (specific application): {type_counts.get(data_config.CHAIN_TYPE_A, 0)}")
    print(f"  ✓ Type B (general enabler): {type_counts.get(data_config.CHAIN_TYPE_B, 0)}")
    
    return df


def validate_threshold_files(
    spec_names: List[str],
    other_names: List[str]
) -> Dict:
    """
    Validate threshold files.
    
    Args:
        spec_names: List of specific chain names
        other_names: List of "other" chain names
    
    Returns:
        Dictionary with threshold data
    """
    print("Validating threshold files...")
    
    # Validate T_L0
    if not data_config.THRESHOLD_L0_PATH.exists():
        raise FileNotFoundError(
            f"T_L0 file not found: {data_config.THRESHOLD_L0_PATH}"
        )
    
    t_l0_data = load_json(data_config.THRESHOLD_L0_PATH)
    if 'T_L0' not in t_l0_data:
        raise ValueError("Missing 'T_L0' key in threshold_l0.json")
    
    t_l0 = float(t_l0_data['T_L0'])
    print(f"  ✓ T_L0: {t_l0}")
    
    # Validate Threshold_Spec
    if not data_config.THRESHOLD_SPEC_PATH.exists():
        raise FileNotFoundError(
            f"Threshold_Spec file not found: {data_config.THRESHOLD_SPEC_PATH}"
        )
    
    df_spec = load_csv(data_config.THRESHOLD_SPEC_PATH)
    if 'chain_name' not in df_spec.columns or 'threshold' not in df_spec.columns:
        raise ValueError("Threshold_Spec file must have 'chain_name' and 'threshold' columns")
    
    # Check all specific chains have thresholds
    spec_set = set(spec_names)
    thresh_spec_set = set(df_spec['chain_name'])
    
    missing = spec_set - thresh_spec_set
    if missing:
        raise ValueError(
            f"{len(missing)} specific chains missing from Threshold_Spec: {list(missing)[:5]}..."
        )
    
    print(f"  ✓ Threshold_Spec covers all {len(spec_names)} specific chains")
    
    # Validate Threshold_Other
    if not data_config.THRESHOLD_OTHER_PATH.exists():
        raise FileNotFoundError(
            f"Threshold_Other file not found: {data_config.THRESHOLD_OTHER_PATH}"
        )
    
    df_other = load_csv(data_config.THRESHOLD_OTHER_PATH)
    if 'chain_name' not in df_other.columns or 'threshold' not in df_other.columns:
        raise ValueError("Threshold_Other file must have 'chain_name' and 'threshold' columns")
    
    # Check all "other" chains have thresholds
    other_set = set(other_names)
    thresh_other_set = set(df_other['chain_name'])
    
    missing = other_set - thresh_other_set
    if missing:
        raise ValueError(
            f"{len(missing)} 'other' chains missing from Threshold_Other: {list(missing)[:5]}..."
        )
    
    print(f"  ✓ Threshold_Other covers all {len(other_names)} 'other' chains")
    
    return {
        't_l0': t_l0,
        'threshold_spec_df': df_spec,
        'threshold_other_df': df_other
    }


def validate_input_files() -> Dict:
    """
    Validate all input files.
    
    Returns:
        Dictionary with all validation results
    """
    print("=" * 70)
    print("VALIDATING INPUT FILES")
    print("=" * 70)
    print()
    
    results = {}
    
    # Validate L0 embeddings
    l0_data = validate_l0_embeddings()
    results['l0_data'] = l0_data
    print()
    
    # Validate chain embeddings
    chain_data = validate_chain_embeddings()
    results.update(chain_data)
    print()
    
    # 验证源文本嵌入向量
    source_data = validate_source_embeddings()
    results['source_data'] = source_data
    print()

    # 验证嵌入向量维度一致性
    if l0_data['d'] != chain_data['d']:
        raise ValueError(
            f"Embedding dimension mismatch: "
            f"L0 has d={l0_data['d']}, "
            f"chains have d={chain_data['d']}"
        )

    if chain_data['d'] != source_data['d']:
        raise ValueError(
            f"Embedding dimension mismatch: "
            f"chains have d={chain_data['d']}, "
            f"sources have d={source_data['d']}"
        )
    
    print("=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    validate_input_files()

