#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
产业链环节定义向量化脚本

使用embedding模型对产业链定义数据进行向量化处理，并保存为npz格式。
"""

import os
import sys
import argparse
import json
from pathlib import Path

import yaml


# ==================== 配置参数 ====================
DEFAULT_MODEL_PATH = os.getenv("EMBEDDING_MODEL_PATH", "/path/to/Qwen3-Embedding-8B")
DEFAULT_L0_OUTPUT = 'l0_embedding.npz'
DEFAULT_CHAIN_OUTPUT = 'chain_embeddings.npz'
DEFAULT_CLASSIFICATION_OUTPUT = 'chain_type_classification.csv'
DEFAULT_BATCH_SIZE = 256
DEFAULT_CUDA_DEVICE = 0
DEFAULT_CACHE_CLEAR_INTERVAL = 100
DEFAULT_PREFIX_TEMPLATE = '哪些文本描述满足产业链环节"{chain_name}"的定义：'

# 动态路径配置
_script_dir = Path(__file__).parent
DEFAULT_INPUT_FILE = str(_script_dir / 'chain_definitions_formatted/chain_definitions_示例产业链_list.json')
DEFAULT_OUTPUT_DIR = str(_script_dir / 'chain_definitions_embedded')


def load_config():
    """
    从配置文件加载脚本的默认参数配置

    Returns:
        dict: 包含 embed_chain_definitions 脚本配置的字典，如果加载失败则返回空字典
    """
    config_file = Path(__file__).parent / 'config.yaml'

    if not config_file.exists():
        print(f"警告: 配置文件不存在: {config_file}")
        return {}

    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            if config and 'embed_chain_definitions' in config:
                return config['embed_chain_definitions']
            else:
                print(f"警告: 配置文件中未找到 embed_chain_definitions 配置")
                return {}
    except (yaml.YAMLError, IOError) as e:
        print(f"警告: 无法读取配置文件: {str(e)}")
        return {}


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
    # 加载配置文件中的默认值
    config = load_config()

    # 获取脚本所在目录
    script_dir = Path(__file__).parent

    # 处理 input_file 和 output_dir 的默认值
    default_input_file = str(script_dir / config.get('input_file', 'chain_definitions_formatted/chain_definitions_示例产业链_list.json'))
    default_output_dir = str(script_dir / config.get('output_dir', 'chain_definitions_embedded'))

    parser = argparse.ArgumentParser(
        description='产业链环节定义向量化脚本',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--input_file',
        type=str,
        default=default_input_file,
        help='输入JSON文件路径'
    )

    parser.add_argument(
        '--model_path',
        type=str,
        default=config.get('model_path', DEFAULT_MODEL_PATH),
        help='Embedding模型路径'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default=default_output_dir,
        help='输出目录路径'
    )

    parser.add_argument(
        '--l0_output',
        type=str,
        default=config.get('l0_output', DEFAULT_L0_OUTPUT),
        help='主产业链名称embeddings输出文件名'
    )

    parser.add_argument(
        '--chain_output',
        type=str,
        default=config.get('chain_output', DEFAULT_CHAIN_OUTPUT),
        help='产业链环节链路embeddings输出文件名'
    )

    parser.add_argument(
        '--classification_output',
        type=str,
        default=config.get('classification_output', DEFAULT_CLASSIFICATION_OUTPUT),
        help='产业链环节分类CSV输出文件名'
    )

    parser.add_argument(
        '--definition_prefix',
        type=str,
        default=config.get('definition_prefix', DEFAULT_PREFIX_TEMPLATE),
        help='添加到每个定义前的前缀模板，使用{chain_name}作为占位符'
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=config.get('batch_size', DEFAULT_BATCH_SIZE),
        help='批处理大小'
    )

    parser.add_argument(
        '--cuda_device',
        type=int,
        default=config.get('cuda_device', DEFAULT_CUDA_DEVICE),
        help='CUDA设备ID'
    )

    parser.add_argument(
        '--cache_clear_interval',
        type=int,
        default=config.get('cache_clear_interval', DEFAULT_CACHE_CLEAR_INTERVAL),
        help='GPU缓存清理间隔（每N个batch清空一次）'
    )

    return parser.parse_args()


# 解析参数并设置CUDA设备（必须在导入torch之前）
args = parse_args()
setup_cuda_device(args.cuda_device)

# 现在可以安全地导入torch及相关库
import torch
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict

sys.path.insert(0, str(Path(__file__).parent))
from embed_utils import load_embedding_model, generate_embeddings


def load_json_data(input_file: str) -> List[Dict]:
    """
    加载JSON数据文件
    
    Args:
        input_file: 输入JSON文件路径
        
    Returns:
        list: JSON数据列表
    """
    print(f"正在读取数据: {input_file}")
    
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"文件不存在: {input_file}")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"  读取 {len(data):,} 条记录")
    
    return data


def filter_data_by_type(data: List[Dict], target_type: str) -> Tuple[List[str], List[str], List[str]]:
    """
    根据type字段过滤数据
    
    Args:
        data: JSON数据列表
        target_type: 目标类型 ("主产业链名称" 或 "产业链环节链路")
        
    Returns:
        tuple: (names列表, definitions列表, dependency_types列表)
    """
    names = []
    definitions = []
    dependency_types = []
    
    for record in data:
        if record.get('type') == target_type:
            names.append(record.get('name', ''))
            definitions.append(record.get('definition', ''))
            dependency_types.append(record.get('dependency_type', ''))
    
    print(f"  过滤类型 '{target_type}': {len(names)} 条记录")
    
    return names, definitions, dependency_types


def apply_prefix(names: List[str], definitions: List[str], prefix_template: str) -> List[str]:
    """
    为定义文本添加前缀
    
    Args:
        names: 名称列表
        definitions: 定义文本列表
        prefix_template: 前缀模板（可包含{chain_name}占位符）
        
    Returns:
        list: 添加前缀后的文本列表
    """
    if not prefix_template:
        return definitions
    
    prefixed_definitions = []
    for name, definition in zip(names, definitions):
        # 替换模板中的{chain_name}占位符
        prefix = prefix_template.format(chain_name=name)
        prefixed_definitions.append(prefix + definition)
    
    return prefixed_definitions


def save_embeddings(names: List[str], embeddings: np.ndarray, output_file: str):
    """
    保存embeddings为npz格式
    
    Args:
        names: 名称列表
        embeddings: embeddings数组
        output_file: 输出文件路径
    """
    print(f"正在保存结果: {output_file}")
    
    output_dir = os.path.dirname(output_file)
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    names_array = np.array(names)
    np.savez(output_file, chain_names=names_array, embeddings=embeddings)
    
    file_size = os.path.getsize(output_file) / (1024**2)
    print(f"  保存完成，文件大小: {file_size:.2f} MB")


def save_classification_csv(names: List[str], dependency_types: List[str], output_file: str):
    """
    保存产业链环节分类CSV文件
    
    Args:
        names: 名称列表
        dependency_types: 依赖类型列表
        output_file: 输出文件路径
    """
    print(f"正在保存分类文件: {output_file}")
    
    output_dir = os.path.dirname(output_file)
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 将dependency_type映射为A或B
    type_mapping = []
    for dt in dependency_types:
        if dt == 'Specific':
            type_mapping.append('A')
        elif dt == 'General':
            type_mapping.append('B')
        else:
            type_mapping.append('')  # 处理未知类型
    
    df_classification = pd.DataFrame({
        'chain_name': names,
        'type': type_mapping
    })
    
    df_classification.to_csv(output_file, index=False, encoding='utf-8')
    print(f"  保存完成，共 {len(df_classification)} 条记录")


def verify_saved_data(output_file: str):
    """
    验证保存的数据
    
    Args:
        output_file: 输出文件路径
    """
    loaded = np.load(output_file, allow_pickle=True)
    print(f"  验证: chain_names={len(loaded['chain_names']):,}, embeddings={loaded['embeddings'].shape}")


def main():
    """主函数"""
    if not torch.cuda.is_available():
        print("警告: CUDA不可用，脚本可能无法正常运行")
        sys.exit(1)
    
    print(f"使用GPU: cuda:0 (物理设备 {args.cuda_device})")
    print(f"定义前缀: '{args.definition_prefix}'" if args.definition_prefix else "定义前缀: (空)")
    print("=" * 70)
    
    # 1. 加载模型
    model = load_embedding_model(args.model_path)
    print("=" * 70)
    
    # 2. 加载JSON数据
    data = load_json_data(args.input_file)
    print("=" * 70)
    
    # 3. 处理主产业链名称数据 (type="主产业链名称")
    print("处理主产业链名称数据...")
    l0_names, l0_definitions, _ = filter_data_by_type(data, "主产业链名称")
    
    if l0_names:
        l0_definitions_prefixed = apply_prefix(l0_names, l0_definitions, args.definition_prefix)
        l0_embeddings = generate_embeddings(
            model,
            l0_definitions_prefixed,
            prompt_name="query",
            batch_size=args.batch_size,
            cache_clear_interval=args.cache_clear_interval
        )
        
        l0_output_file = os.path.join(args.output_dir, args.l0_output)
        save_embeddings(l0_names, l0_embeddings, l0_output_file)
        verify_saved_data(l0_output_file)
    else:
        print("  警告: 未找到主产业链名称数据")
    
    print("=" * 70)
    
    # 4. 处理产业链环节链路数据 (type="产业链环节链路")
    print("处理产业链环节链路数据...")
    chain_names, chain_definitions, dependency_types = filter_data_by_type(data, "产业链环节链路")
    
    if chain_names:
        chain_definitions_prefixed = apply_prefix(chain_names, chain_definitions, args.definition_prefix)
        chain_embeddings = generate_embeddings(
            model,
            chain_definitions_prefixed,
            prompt_name="query",
            batch_size=args.batch_size,
            cache_clear_interval=args.cache_clear_interval
        )
        
        chain_output_file = os.path.join(args.output_dir, args.chain_output)
        save_embeddings(chain_names, chain_embeddings, chain_output_file)
        verify_saved_data(chain_output_file)
        
        # 5. 保存分类CSV文件
        classification_output_file = os.path.join(args.output_dir, args.classification_output)
        save_classification_csv(chain_names, dependency_types, classification_output_file)
    else:
        print("  警告: 未找到产业链环节链路数据")
    
    print("=" * 70)
    print("处理完成")


if __name__ == '__main__':
    main()

