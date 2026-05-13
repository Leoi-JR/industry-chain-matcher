#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
源文本数据预处理脚本

该脚本对原始源文本数据进行清洗和预处理，包括：
1. 标点符号标准化
2. 高频括号内容过滤
3. 特殊字符清理
4. 数据筛选和排序
5. 分割输出为多个文件

处理后的数据将用于后续的相似度计算等操作。
"""

import glob
import pandas as pd
import re
import os
import gc
import argparse
from collections import Counter
from typing import List
from tqdm import tqdm


def load_and_merge_data(input_pattern: str) -> pd.DataFrame:
    """
    加载并合并所有parquet文件
    
    Args:
        input_pattern: 输入文件的匹配模式，如 'source_texts/*.parquet'
        
    Returns:
        pd.DataFrame: 合并后的数据框
    """
    print(f"正在加载数据文件: {input_pattern}")
    parquet_files = glob.glob(input_pattern)
    
    if not parquet_files:
        raise FileNotFoundError(f"未找到匹配的文件: {input_pattern}")
    
    print(f"  找到 {len(parquet_files)} 个文件")
    
    dfs = [pd.read_parquet(f) for f in parquet_files]
    df = pd.concat(dfs, ignore_index=True)
    
    print(f"  合并后总记录数: {len(df):,}")
    
    return df


def normalize_punctuation(df: pd.DataFrame) -> pd.DataFrame:
    """
    标准化标点符号
    
    将半角括号转换为全角括号，删除特殊符号
    
    Args:
        df: 输入数据框
        
    Returns:
        pd.DataFrame: 标准化后的数据框
    """
    print("\n正在标准化标点符号...")
    
    df['source_text'] = (
        df['source_text']
        .str.replace('(', '（', regex=False)
        .str.replace(')', '）', regex=False)
        .str.replace('*', '', regex=False)
        .str.replace('^', '', regex=False)
    )
    
    print("  标点符号标准化完成")
    
    return df


def extract_parentheses_content(text: str) -> List[str]:
    """
    提取文本中所有被全角括号包裹的内容
    
    匹配优先级: 非嵌套的（）内容，且包括左右括号
    
    Args:
        text: 输入文本
        
    Returns:
        list: 括号内容列表（包含括号本身）
    """
    pattern = r'（[^（）]*?）'
    return re.findall(pattern, text) if isinstance(text, str) else []


def identify_high_freq_patterns(df: pd.DataFrame, freq_threshold: int) -> List[str]:
    """
    识别高频出现的括号内容
    
    Args:
        df: 输入数据框
        freq_threshold: 频次阈值，超过此值的内容将被识别为高频
        
    Returns:
        list: 高频括号内容列表
    """
    print(f"\n正在识别高频括号内容（阈值: {freq_threshold:,}）...")
    
    # 提取所有括号内容
    all_par_list = df['source_text'].apply(extract_parentheses_content)
    paren_contents = [item for sublist in all_par_list for item in sublist]
    
    print(f"  提取到 {len(paren_contents):,} 个括号内容")
    
    # 统计频次
    paren_counter = Counter(paren_contents)
    sorted_paren_counts = paren_counter.most_common()
    
    # 获取高频内容
    high_freq_paren = [value for value, cnt in sorted_paren_counts if cnt > freq_threshold]
    
    print(f"  识别到 {len(high_freq_paren)} 个高频括号内容")
    if high_freq_paren:
        print(f"  示例高频内容（前3个）:")
        for i, paren in enumerate(high_freq_paren[:3], 1):
            count = paren_counter[paren]
            print(f"    {i}. {paren[:30]}... (出现 {count:,} 次)")
    
    # 释放内存
    del all_par_list
    del paren_contents
    del paren_counter
    del sorted_paren_counts
    gc.collect()
    
    return high_freq_paren


def remove_patterns(df: pd.DataFrame, patterns: List[str]) -> pd.DataFrame:
    """
    从文本中删除指定的模式
    
    Args:
        df: 输入数据框
        patterns: 要删除的模式列表
        
    Returns:
        pd.DataFrame: 处理后的数据框
    """
    if not patterns:
        print("\n没有需要删除的模式")
        return df
    
    print(f"\n正在删除 {len(patterns)} 个高频括号内容...")
    
    def remove_high_freq_paren(text):
        if not isinstance(text, str):
            return text
        for paren in patterns:
            text = text.replace(paren, '')
        return text
    
    df['source_text'] = df['source_text'].apply(remove_high_freq_paren)
    
    print("  高频内容删除完成")
    
    return df


def clean_special_chars(df: pd.DataFrame) -> pd.DataFrame:
    """
    清理特殊字符
    
    删除问号、星号等特殊字符（包括全角和半角）
    
    Args:
        df: 输入数据框
        
    Returns:
        pd.DataFrame: 清理后的数据框
    """
    print("\n正在清理特殊字符...")
    
    df['source_text'] = (
        df['source_text']
        .str.replace('?', '', regex=False)
        .str.replace('*', '', regex=False)
        .str.replace('？', '', regex=False)
        .str.replace('＊', '', regex=False)
    )
    
    print("  特殊字符清理完成")
    
    return df


def filter_and_sort(df: pd.DataFrame, min_length: int) -> pd.DataFrame:
    """
    按长度过滤和排序数据
    
    Args:
        df: 输入数据框
        min_length: 最小文本长度
        
    Returns:
        pd.DataFrame: 过滤和排序后的数据框
    """
    print(f"\n正在按长度过滤和排序（最小长度: {min_length}）...")
    
    original_count = len(df)
    
    # 按长度降序排序
    df = df.sort_values(by='source_text', key=lambda x: x.str.len(), ascending=False)
    
    # 过滤长度小于阈值的记录
    df = df[df['source_text'].str.len() >= min_length]
    
    filtered_count = len(df)
    removed_count = original_count - filtered_count
    
    print(f"  排序完成")
    print(f"  过滤前记录数: {original_count:,}")
    print(f"  过滤后记录数: {filtered_count:,}")
    print(f"  删除记录数: {removed_count:,}")
    
    if filtered_count > 0:
        max_len = df['source_text'].str.len().max()
        min_len = df['source_text'].str.len().min()
        avg_len = df['source_text'].str.len().mean()
        print(f"  文本长度统计: 最大={max_len}, 最小={min_len}, 平均={avg_len:.1f}")
    
    return df


def split_and_save(df: pd.DataFrame, output_dir: str, num_splits: int):
    """
    将数据分割并保存为多个parquet文件
    
    Args:
        df: 输入数据框
        output_dir: 输出目录
        num_splits: 分割文件数量
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n正在分割数据并保存到: {output_dir}")
    print(f"  分割文件数: {num_splits}")
    
    chunk_size = len(df) // num_splits
    print(f"  每个文件约 {chunk_size:,} 条记录")
    
    for i in tqdm(range(num_splits), desc="保存文件"):
        start_idx = i * chunk_size
        # 确保最后一个文件包含所有剩余记录
        end_idx = (i + 1) * chunk_size if i != num_splits - 1 else len(df)
        chunk = df.iloc[start_idx:end_idx]
        
        output_file = f'{output_dir}/source_part_{i+1}.parquet'
        chunk.to_parquet(output_file, index=False)
    
    print(f"  保存完成! 共生成 {num_splits} 个文件")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='Source text preprocessing script',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--input_pattern',
        type=str,
        default='source_texts/*.parquet',
        help='输入文件的匹配模式'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='source_texts_split',
        help='输出目录路径'
    )
    
    parser.add_argument(
        '--freq_threshold',
        type=int,
        default=100000,
        help='高频括号内容的频次阈值'
    )
    
    parser.add_argument(
        '--min_length',
        type=int,
        default=20,
        help='文本的最小长度'
    )
    
    parser.add_argument(
        '--num_splits',
        type=int,
        default=60,
        help='输出文件的分割数量'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("源文本数据预处理")
    print("=" * 60)
    
    # 1. 加载并合并数据
    df = load_and_merge_data(args.input_pattern)
    
    # 2. 标准化标点符号
    df = normalize_punctuation(df)
    
    # 3. 识别高频括号内容
    high_freq_patterns = identify_high_freq_patterns(df, args.freq_threshold)
    
    # 4. 删除高频括号内容
    df = remove_patterns(df, high_freq_patterns)
    
    # 5. 清理特殊字符
    df = clean_special_chars(df)
    
    # 6. 过滤和排序
    df = filter_and_sort(df, args.min_length)
    
    # 7. 分割并保存
    split_and_save(df, args.output_dir, args.num_splits)
    
    print("\n" + "=" * 60)
    print("处理完成!")
    print("=" * 60)


if __name__ == '__main__':
    main()

