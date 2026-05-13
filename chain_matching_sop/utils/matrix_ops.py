#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Matrix operations for chain matching SOP
"""

import numpy as np
from typing import Tuple, Optional

# Try to import CuPy for GPU acceleration
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None


def is_gpu_available() -> bool:
    """
    Check if GPU (CUDA) is available for computation.
    
    Returns:
        True if CuPy is available and CUDA is functional
    """
    if not CUPY_AVAILABLE:
        return False
    
    try:
        # Try to access CUDA device
        _ = cp.cuda.Device(0)
        return True
    except Exception:
        return False


def get_gpu_info() -> dict:
    """
    Get GPU device information.
    
    Returns:
        Dictionary with GPU information or empty dict if not available
    """
    if not is_gpu_available():
        return {}
    
    try:
        device = cp.cuda.Device()
        mem_info = device.mem_info
        return {
            'device_id': device.id,
            'device_name': cp.cuda.runtime.getDeviceProperties(device.id)['name'].decode(),
            'total_memory_gb': mem_info[1] / (1024**3),
            'free_memory_gb': mem_info[0] / (1024**3)
        }
    except Exception as e:
        return {'error': str(e)}


def normalize_embeddings(embeddings: np.ndarray, use_gpu: bool = False) -> np.ndarray:
    """
    Normalize embeddings to unit length (L2 normalization).
    
    Args:
        embeddings: Array of shape (n, d) where n is number of vectors,
                   d is embedding dimension
        use_gpu: Whether to use GPU acceleration
    
    Returns:
        Normalized embeddings of same shape (as NumPy array)
    """
    if use_gpu and is_gpu_available():
        # GPU computation
        embeddings_gpu = cp.asarray(embeddings)
        norms = cp.linalg.norm(embeddings_gpu, axis=1, keepdims=True)
        norms = cp.where(norms == 0, 1, norms)
        normalized = embeddings_gpu / norms
        return cp.asnumpy(normalized)
    else:
        # CPU computation
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        normalized = embeddings / norms
        return normalized


def compute_cosine_similarity(
    embeddings_a: np.ndarray,
    embeddings_b: np.ndarray,
    normalize: bool = True,
    use_gpu: bool = False
) -> np.ndarray:
    """
    Compute cosine similarity between two sets of embeddings.
    
    Args:
        embeddings_a: Array of shape (m, d)
        embeddings_b: Array of shape (n, d)
        normalize: Whether to normalize embeddings first
        use_gpu: Whether to use GPU acceleration
    
    Returns:
        Similarity matrix of shape (m, n) (as NumPy array)
    """
    if use_gpu and is_gpu_available():
        # GPU computation
        # Transfer to GPU
        embeddings_a_gpu = cp.asarray(embeddings_a)
        embeddings_b_gpu = cp.asarray(embeddings_b)
        
        if normalize:
            # Normalize on GPU
            norms_a = cp.linalg.norm(embeddings_a_gpu, axis=1, keepdims=True)
            norms_a = cp.where(norms_a == 0, 1, norms_a)
            embeddings_a_gpu = embeddings_a_gpu / norms_a
            
            norms_b = cp.linalg.norm(embeddings_b_gpu, axis=1, keepdims=True)
            norms_b = cp.where(norms_b == 0, 1, norms_b)
            embeddings_b_gpu = embeddings_b_gpu / norms_b
        
        # Compute dot product on GPU: (m, d) @ (d, n) = (m, n)
        similarity_gpu = embeddings_a_gpu @ embeddings_b_gpu.T
        
        # Clip to [-1, 1] to handle numerical errors
        similarity_gpu = cp.clip(similarity_gpu, -1.0, 1.0)
        
        # Transfer back to CPU
        return cp.asnumpy(similarity_gpu)
    else:
        # CPU computation
        if normalize:
            embeddings_a = normalize_embeddings(embeddings_a, use_gpu=False)
            embeddings_b = normalize_embeddings(embeddings_b, use_gpu=False)
        
        # Compute dot product: (m, d) @ (d, n) = (m, n)
        similarity = embeddings_a @ embeddings_b.T
        
        # Clip to [-1, 1] to handle numerical errors
        similarity = np.clip(similarity, -1.0, 1.0)
        
        return similarity


def apply_threshold(
    similarity: np.ndarray,
    threshold: np.ndarray
) -> np.ndarray:
    """
    Apply threshold to similarity matrix to create binary mask.
    
    Args:
        similarity: Similarity matrix of shape (m, n)
        threshold: Threshold vector of shape (m, 1) or scalar
    
    Returns:
        Binary mask of shape (m, n) where True indicates similarity > threshold
    """
    # Ensure threshold has correct shape for broadcasting
    if threshold.ndim == 1:
        threshold = threshold.reshape(-1, 1)
    
    mask = similarity > threshold
    
    return mask


def compute_cascade_mask(
    mask_l0: np.ndarray,
    chain_type_vector: np.ndarray
) -> np.ndarray:
    """
    Compute cascade mask based on chain type.

    For A-type chains (type=1): result = mask_l0
    For B-type chains (type=0): result = 1 (always pass)

    Formula: (mask_l0 * type) + (1 - type)

    Args:
        mask_l0: L0 mask of shape (m, n) - boolean array
        chain_type_vector: Type vector of shape (m, 1) with values 0 or 1

    Returns:
        Cascade mask of shape (m, n) - boolean array
    """
    # 确保类型向量具有正确的形状
    if chain_type_vector.ndim == 1:
        chain_type_vector = chain_type_vector.reshape(-1, 1)

    # 确保输入是布尔类型
    mask_l0 = mask_l0.astype(bool)

    # 将类型向量转换为布尔类型以进行逻辑运算
    type_is_a = chain_type_vector.astype(bool)  # True for A-type (1), False for B-type (0)

    # 对于A类型链条：结果 = mask_l0
    # 对于B类型链条：结果 = True (总是通过)
    # 使用逻辑运算：(mask_l0 & type_is_a) | (~type_is_a)
    cascade_mask = (mask_l0 & type_is_a) | (~type_is_a)

    return cascade_mask


def compute_exclusion_mask(
    mask_spec: np.ndarray,
    exclusion_matrix: np.ndarray
) -> np.ndarray:
    """
    Compute exclusion mask for "other" chains.
    
    Args:
        mask_spec: Specific chain mask of shape (m_spec, n)
        exclusion_matrix: Exclusion mapping of shape (m_other, m_spec)
    
    Returns:
        No_Sibling_Match mask of shape (m_other, n)
        True indicates source did NOT match any sibling
    """
    # Convert mask to float for matrix multiplication
    mask_spec_float = mask_spec.astype(float)
    
    # Compute how many siblings each source matched
    # (m_other, m_spec) @ (m_spec, n) = (m_other, n)
    sibling_match_count = exclusion_matrix @ mask_spec_float
    
    # Any_Sibling_Match: True if count > 0
    any_sibling_match = sibling_match_count > 0
    
    # No_Sibling_Match: invert
    no_sibling_match = ~any_sibling_match
    
    return no_sibling_match


def assemble_total_matrix(
    indices: np.ndarray,
    partial_matrix: np.ndarray,
    m_total: int,
    n: int
) -> np.ndarray:
    """
    Assemble a partial matrix into a total matrix using indices.
    
    Args:
        indices: Array of row indices where to place partial matrix
        partial_matrix: Partial matrix of shape (m_partial, n)
        m_total: Total number of rows in output
        n: Number of columns
    
    Returns:
        Total matrix of shape (m_total, n)
    """
    # Initialize zero matrix
    total_matrix = np.zeros((m_total, n), dtype=partial_matrix.dtype)
    
    # Place partial matrix at specified indices
    total_matrix[indices, :] = partial_matrix
    
    return total_matrix

