#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
源文本向量化脚本

使用 embedding 模型对任意源文本（如公司描述、文档摘要等）进行向量化，保存为 npz 格式。
输入 parquet 文件须包含 ID 列和文本列，列名通过 --id_column / --text_column 指定。
"""

import os
import sys
import argparse


# ==================== 配置参数（方便开发者修改默认值） ====================
DEFAULT_MODEL_PATH = os.getenv("EMBEDDING_MODEL_PATH", "/path/to/Qwen3-Embedding-8B")
DEFAULT_OUTPUT_DIR = 'source_text_embeddings'
DEFAULT_ID_COLUMN = 'id'
DEFAULT_TEXT_COLUMN = 'source_text'
DEFAULT_BATCH_SIZE = 256
DEFAULT_CUDA_DEVICE = 0
DEFAULT_CACHE_CLEAR_INTERVAL = 100


def setup_cuda_device(device_id: int):
    """
    设置CUDA设备
    
    必须在导入torch之前调用，否则设置无效。
    
    Args:
        device_id: CUDA设备ID（物理设备编号）
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = str(device_id)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='源文本向量化脚本',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--input_file',
        type=str,
        required=True,
        help='输入parquet文件路径'
    )
    
    parser.add_argument(
        '--model_path',
        type=str,
        default=DEFAULT_MODEL_PATH,
        help='Embedding模型路径'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help='输出目录路径'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help='批处理大小'
    )
    
    parser.add_argument(
        '--cuda_device',
        type=int,
        default=DEFAULT_CUDA_DEVICE,
        help='CUDA设备ID'
    )
    
    parser.add_argument(
        '--cache_clear_interval',
        type=int,
        default=DEFAULT_CACHE_CLEAR_INTERVAL,
        help='GPU缓存清理间隔（每N个batch清空一次）'
    )

    parser.add_argument(
        '--id_column',
        type=str,
        default=DEFAULT_ID_COLUMN,
        help='parquet 文件中 ID 列的列名'
    )

    parser.add_argument(
        '--text_column',
        type=str,
        default=DEFAULT_TEXT_COLUMN,
        help='parquet 文件中待向量化的文本列列名（如 source_text 等）'
    )
    
    return parser.parse_args()


# 解析参数并设置CUDA设备（必须在导入torch之前）
args = parse_args()
setup_cuda_device(args.cuda_device)

# 现在可以安全地导入torch及相关库
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List

sys.path.insert(0, str(Path(__file__).parent))
from embed_utils import load_embedding_model, generate_embeddings


def load_data(input_file: str, id_column: str, text_column: str) -> Tuple[List, List]:
    """
    加载数据文件

    Args:
        input_file: 输入parquet文件路径
        id_column: ID列的列名
        text_column: 文本列的列名

    Returns:
        tuple: (ids列表, texts列表)
    """
    print(f"正在读取数据: {input_file}")

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"文件不存在: {input_file}")

    df = pd.read_parquet(input_file)
    print(f"  读取 {len(df):,} 条记录")

    ids = df[id_column].tolist()
    scopes = df[text_column].tolist()

    return ids, scopes


def save_embeddings(ids: List, embeddings: np.ndarray, output_file: str):
    """
    保存embeddings为npz格式
    
    Args:
        ids: ID列表
        embeddings: embeddings数组
        output_file: 输出文件路径
    """
    print(f"正在保存结果: {output_file}")
    
    output_dir = os.path.dirname(output_file)
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    ids_array = np.array(ids)
    np.savez(output_file, ids=ids_array, embeddings=embeddings)
    
    file_size = os.path.getsize(output_file) / (1024**2)
    print(f"  保存完成，文件大小: {file_size:.2f} MB")


def verify_saved_data(output_file: str):
    """
    验证保存的数据
    
    Args:
        output_file: 输出文件路径
    """
    loaded = np.load(output_file)
    print(f"  验证: IDs={len(loaded['ids']):,}, Embeddings={loaded['embeddings'].shape}")


def main():
    """主函数"""
    if not torch.cuda.is_available():
        print("警告: CUDA不可用，脚本可能无法正常运行")
        sys.exit(1)
    
    print(f"使用GPU: cuda:0 (物理设备 {args.cuda_device})")
    
    # 生成输出文件路径
    input_filename = os.path.basename(args.input_file)
    output_filename = input_filename.replace('.parquet', '_embeddings.npz')
    output_file = os.path.join(args.output_dir, output_filename)
    
    # 1. 加载模型
    model = load_embedding_model(args.model_path)
    
    # 2. 加载数据
    ids, scopes = load_data(args.input_file, args.id_column, args.text_column)
    
    # 3. 生成embeddings
    embeddings = generate_embeddings(
        model,
        scopes,
        batch_size=args.batch_size,
        cache_clear_interval=args.cache_clear_interval
    )
    
    # 4. 保存结果
    save_embeddings(ids, embeddings, output_file)
    
    # 5. 验证
    verify_saved_data(output_file)
    
    print("处理完成")


if __name__ == '__main__':
    main()

