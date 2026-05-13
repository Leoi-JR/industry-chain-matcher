#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared embedding utilities for chain definition and source text vectorization.
"""

import os
import numpy as np
import torch
from tqdm import tqdm
from typing import List, Optional
from sentence_transformers import SentenceTransformer


def load_embedding_model(model_path: str) -> SentenceTransformer:
    """
    Load SentenceTransformer with float16 + flash_attention_2 optimizations.

    Args:
        model_path: Local path to the embedding model.

    Returns:
        Loaded SentenceTransformer model.
    """
    print(f"正在加载模型: {model_path}")

    model = SentenceTransformer(
        model_path,
        model_kwargs={
            "attn_implementation": "flash_attention_2",
            "torch_dtype": torch.float16,
            "device_map": "auto",
        },
        tokenizer_kwargs={"padding_side": "left"},
    )

    print("  模型加载完成，设备: cuda:0")
    return model


def generate_embeddings(
    model: SentenceTransformer,
    texts: List[str],
    prompt_name: Optional[str] = None,
    batch_size: int = 256,
    cache_clear_interval: int = 100,
) -> np.ndarray:
    """
    Batch-encode texts with periodic GPU cache management.

    The ``prompt_name`` parameter controls the Qwen3-Embedding instruction mode:
    - Pass ``prompt_name="query"`` for chain definition texts (query side).
    - Leave ``None`` for source texts (passage side, no instruction).

    Args:
        model: Loaded SentenceTransformer model.
        texts: List of strings to encode.
        prompt_name: Optional prompt name passed to ``model.encode``.
        batch_size: Number of texts per GPU batch.
        cache_clear_interval: Clear GPU cache every N batches.

    Returns:
        Float32 numpy array of shape (len(texts), embedding_dim).
    """
    print(f"正在生成embeddings (batch_size={batch_size})...")

    embeddings = []

    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="处理进度"):
            batch = texts[i : i + batch_size]

            encode_kwargs = dict(
                show_progress_bar=False,
                device="cuda:0",
                convert_to_numpy=True,
            )
            if prompt_name is not None:
                encode_kwargs["prompt_name"] = prompt_name

            batch_embeddings = model.encode(batch, **encode_kwargs)

            if isinstance(batch_embeddings, torch.Tensor):
                batch_embeddings = batch_embeddings.cpu().numpy()

            embeddings.append(batch_embeddings)

            if (i // batch_size) % cache_clear_interval == 0 and i > 0:
                torch.cuda.empty_cache()

    embeddings = np.vstack(embeddings)
    print(f"  生成完成，形状: {embeddings.shape}")
    return embeddings
