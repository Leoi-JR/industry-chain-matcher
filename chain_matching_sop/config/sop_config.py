#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SOP configuration parameters
Defines algorithmic parameters and execution settings
"""

# ============================================================================
# Matrix Dimensions (Auto-detected during data preparation)
# ============================================================================

# These will be determined from input data:
# - m_total: Total number of chains
# - m_spec: Number of specific chains
# - m_other: Number of "other" chains
# - d: Embedding dimension
# - n: Total number of sources (sum across all batches)

# ============================================================================
# Execution Settings
# ============================================================================

# Number of decimal places for similarity scores
SIMILARITY_PRECISION = 6

# Whether to normalize embeddings before computing similarity
NORMALIZE_EMBEDDINGS = True

# Batch processing settings
BATCH_PROCESSING = True
MAX_MEMORY_GB = 16  # Maximum memory to use for batch processing

# Progress bar settings
SHOW_PROGRESS = True
PROGRESS_BAR_NCOLS = 100

# ============================================================================
# GPU Settings
# ============================================================================

# Whether to use GPU acceleration (if available)
USE_GPU = True

# GPU device ID (None = auto-select, integer = specific GPU)
# This can be overridden via command-line arguments
GPU_DEVICE_ID = None

# GPU memory fraction (reserved for future use)
GPU_MEMORY_FRACTION = 0.95

# ============================================================================
# Validation Settings
# ============================================================================

# Whether to validate dimensions during execution
VALIDATE_DIMENSIONS = True

# Whether to check for NaN values
CHECK_NAN = True

# Whether to verify parent embeddings exist for all "other" chains
VERIFY_PARENT_EMBEDDINGS = True

# ============================================================================
# Output Settings
# ============================================================================

# Whether to save intermediate results
SAVE_INTERMEDIATE = True

# Compression for numpy files
NPZ_COMPRESSION = True  # Use compressed npz format

# JSON indentation
JSON_INDENT = 2

# CSV settings
CSV_ENCODING = 'utf-8'
CSV_INDEX = False

