#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
I/O utilities for chain matching SOP
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Union


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Ensure directory exists, create if necessary.
    
    Args:
        path: Directory path
    
    Returns:
        Path object
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_npz_file(file_path: Union[str, Path]) -> Dict[str, np.ndarray]:
    """
    Load npz file and return as dictionary.
    
    Args:
        file_path: Path to npz file
    
    Returns:
        Dictionary mapping keys to arrays
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    data = np.load(file_path, allow_pickle=True)
    return {key: data[key] for key in data.keys()}


def save_npz_file(
    file_path: Union[str, Path],
    data: Dict[str, np.ndarray],
    compressed: bool = True
) -> None:
    """
    Save dictionary of arrays to npz file.
    
    Args:
        file_path: Path to save file
        data: Dictionary mapping keys to arrays
        compressed: Whether to use compression
    """
    file_path = Path(file_path)
    ensure_dir(file_path.parent)
    
    if compressed:
        np.savez_compressed(file_path, **data)
    else:
        np.savez(file_path, **data)


def load_json(file_path: Union[str, Path]) -> Any:
    """
    Load JSON file.
    
    Args:
        file_path: Path to JSON file
    
    Returns:
        Loaded JSON data
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(
    file_path: Union[str, Path],
    data: Any,
    indent: int = 2
) -> None:
    """
    Save data to JSON file.
    
    Args:
        file_path: Path to save file
        data: Data to save
        indent: JSON indentation
    """
    file_path = Path(file_path)
    ensure_dir(file_path.parent)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def load_csv(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load CSV file as DataFrame.
    
    Args:
        file_path: Path to CSV file
    
    Returns:
        DataFrame
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    return pd.read_csv(file_path, encoding='utf-8')


def save_csv(
    file_path: Union[str, Path],
    df: pd.DataFrame,
    index: bool = False
) -> None:
    """
    Save DataFrame to CSV file.
    
    Args:
        file_path: Path to save file
        df: DataFrame to save
        index: Whether to save index
    """
    file_path = Path(file_path)
    ensure_dir(file_path.parent)
    
    df.to_csv(file_path, index=index, encoding='utf-8')


def load_npy_file(file_path: Union[str, Path]) -> np.ndarray:
    """
    Load npy file.
    
    Args:
        file_path: Path to npy file
    
    Returns:
        Loaded array
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    return np.load(file_path, allow_pickle=True)


def save_npy_file(
    file_path: Union[str, Path],
    array: np.ndarray
) -> None:
    """
    Save array to npy file.
    
    Args:
        file_path: Path to save file
        array: Array to save
    """
    file_path = Path(file_path)
    ensure_dir(file_path.parent)
    
    np.save(file_path, array)

