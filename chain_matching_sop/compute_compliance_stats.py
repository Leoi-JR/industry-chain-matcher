#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
产业链匹配统计报告生成脚本

功能：
1. 读取 llm_threshold_results.json 文件
2. 统计每个产业链环节在各相似度区间的符合率和样本数
3. 生成 CSV 和 Excel 格式的统计报告表格
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

# 添加当前目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from config.data_config import (
    STATS_INPUT_FILE_PATH,
    STATS_OUTPUT_CSV_PATH,
    STATS_OUTPUT_EXCEL_PATH,
    STATS_MATCH_RESULT_KEY,
    STATS_MISMATCH_RESULT_KEY,
    STATS_INTERVAL_WIDTH,
    THRESHOLD_SAMPLE_COUNT_SUFFIX,
    THRESHOLD_COMPLIANCE_RATE_SUFFIX
)

# 为了保持代码兼容性，创建本地别名
INPUT_FILE_PATH = STATS_INPUT_FILE_PATH
OUTPUT_CSV_PATH = STATS_OUTPUT_CSV_PATH
OUTPUT_EXCEL_PATH = STATS_OUTPUT_EXCEL_PATH
MATCH_RESULT_KEY = STATS_MATCH_RESULT_KEY
MISMATCH_RESULT_KEY = STATS_MISMATCH_RESULT_KEY
INTERVAL_WIDTH = STATS_INTERVAL_WIDTH


def load_json_data(file_path: str) -> Dict:
    """
    加载 JSON 数据文件
    
    Args:
        file_path: JSON 文件路径
        
    Returns:
        解析后的 JSON 数据字典
    """
    print(f"正在读取文件: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"成功读取 {len(data)} 个产业链环节")
    return data


def parse_interval(interval_str: str) -> Tuple[float, float]:
    """
    解析区间字符串，提取下界和上界

    Args:
        interval_str: 区间字符串，格式如 "[0.400, 0.425)"

    Returns:
        (下界, 上界) 元组
    """
    # 移除方括号和圆括号，分割并转换为浮点数
    interval_str = interval_str.strip('[]() ')
    parts = interval_str.split(',')
    lower_bound = float(parts[0].strip())
    upper_bound = float(parts[1].strip())
    return lower_bound, upper_bound


def format_interval(lower_bound: float, upper_bound: float) -> str:
    """
    格式化区间为字符串

    Args:
        lower_bound: 区间下界
        upper_bound: 区间上界

    Returns:
        格式化的区间字符串，格式如 "[0.400, 0.425)"
    """
    return f"[{lower_bound:.3f}, {upper_bound:.3f})"


def standardize_interval(interval_str: str, interval_width: float = INTERVAL_WIDTH) -> str:
    """
    标准化区间：将非标准区间映射到标准区间

    Args:
        interval_str: 原始区间字符串
        interval_width: 标准区间宽度

    Returns:
        标准化后的区间字符串
    """
    lower_bound, upper_bound = parse_interval(interval_str)

    # 计算标准化后的上界：下界 + 区间宽度
    standard_upper_bound = lower_bound + interval_width

    # 返回标准化的区间字符串
    return format_interval(lower_bound, standard_upper_bound)


def normalize_data(data: Dict, interval_width: float = INTERVAL_WIDTH) -> Tuple[Dict, Dict[str, List[str]]]:
    """
    标准化数据：将所有非标准区间映射到标准区间，并合并样本数据

    Args:
        data: 原始 JSON 数据字典
        interval_width: 标准区间宽度

    Returns:
        (标准化后的数据字典, 区间映射关系字典) 元组
        区间映射关系字典的格式为 {标准区间: [原始区间列表]}
    """
    print("\n开始标准化区间...")

    normalized_data = {}
    interval_mapping = {}  # 记录标准区间到原始区间的映射关系
    original_interval_count = 0

    for chain_segment, chain_data in data.items():
        normalized_chain_data = {}

        for original_interval, samples in chain_data.items():
            original_interval_count += 1

            # 标准化区间
            standard_interval = standardize_interval(original_interval, interval_width)

            # 记录映射关系
            if standard_interval not in interval_mapping:
                interval_mapping[standard_interval] = []
            if original_interval not in interval_mapping[standard_interval]:
                interval_mapping[standard_interval].append(original_interval)

            # 合并样本数据到标准区间
            if standard_interval not in normalized_chain_data:
                normalized_chain_data[standard_interval] = []
            normalized_chain_data[standard_interval].extend(samples)

        normalized_data[chain_segment] = normalized_chain_data

    # 统计标准化信息
    standard_interval_count = len(interval_mapping)
    merged_count = sum(1 for intervals in interval_mapping.values() if len(intervals) > 1)

    print(f"原始区间总数: {len(set(interval for chain_data in data.values() for interval in chain_data.keys()))}")
    print(f"标准化后区间总数: {standard_interval_count}")
    print(f"发生合并的区间数: {merged_count}")

    # 显示部分合并示例
    if merged_count > 0:
        print(f"\n区间合并示例（前5个）:")
        count = 0
        for standard_interval, original_intervals in interval_mapping.items():
            if len(original_intervals) > 1:
                print(f"  {standard_interval} <- {original_intervals}")
                count += 1
                if count >= 5:
                    break

    return normalized_data, interval_mapping


def extract_all_intervals(data: Dict) -> List[str]:
    """
    提取所有相似度区间并排序

    Args:
        data: JSON 数据字典

    Returns:
        排序后的相似度区间列表
    """
    all_intervals = set()
    for chain_data in data.values():
        all_intervals.update(chain_data.keys())

    # 按区间起始值排序
    def parse_interval_start(interval_str: str) -> float:
        """解析区间字符串的起始值"""
        return float(interval_str.split(',')[0].strip('['))

    sorted_intervals = sorted(all_intervals, key=parse_interval_start)
    print(f"发现 {len(sorted_intervals)} 个不同的相似度区间")
    return sorted_intervals


def calculate_statistics(samples: List[Dict]) -> Tuple[float, int]:
    """
    计算样本的符合率和样本数
    
    Args:
        samples: 样本列表，每个样本包含 id 和 result 字段
        
    Returns:
        (符合率, 样本总数) 元组
    """
    total_count = len(samples)
    if total_count == 0:
        return 0.0, 0
    
    match_count = sum(1 for sample in samples if sample['result'] == MATCH_RESULT_KEY)
    match_rate = match_count / total_count
    
    return match_rate, total_count


def generate_statistics_table(data: Dict, intervals: List[str]) -> pd.DataFrame:
    """
    生成统计报告表格
    
    Args:
        data: JSON 数据字典
        intervals: 相似度区间列表
        
    Returns:
        统计报告 DataFrame
    """
    print("\n开始生成统计表格...")
    
    # 初始化表格数据
    table_data = []
    
    for chain_segment, chain_data in data.items():
        row = {'产业链环节': chain_segment}
        
        for interval in intervals:
            if interval in chain_data:
                match_rate, sample_count = calculate_statistics(chain_data[interval])
                row[f'{interval}{THRESHOLD_COMPLIANCE_RATE_SUFFIX}'] = match_rate
                row[f'{interval}{THRESHOLD_SAMPLE_COUNT_SUFFIX}'] = sample_count
            else:
                # 该环节不存在此区间，填充空值
                row[f'{interval}{THRESHOLD_COMPLIANCE_RATE_SUFFIX}'] = None
                row[f'{interval}{THRESHOLD_SAMPLE_COUNT_SUFFIX}'] = None
        
        table_data.append(row)
    
    # 创建 DataFrame
    df = pd.DataFrame(table_data)
    
    # 调整列顺序：产业链环节列在最前，然后是各区间的符合率和样本数交替排列
    columns = ['产业链环节']
    for interval in intervals:
        columns.append(f'{interval}{THRESHOLD_COMPLIANCE_RATE_SUFFIX}')
        columns.append(f'{interval}{THRESHOLD_SAMPLE_COUNT_SUFFIX}')
    
    df = df[columns]
    
    print(f"表格生成完成，共 {len(df)} 行，{len(df.columns)} 列")
    return df


def save_to_csv(df: pd.DataFrame, file_path: str) -> None:
    """
    保存 DataFrame 到 CSV 文件
    
    Args:
        df: 统计报告 DataFrame
        file_path: 输出文件路径
    """
    print(f"\n正在保存 CSV 文件: {file_path}")
    df.to_csv(file_path, index=False, encoding='utf-8-sig')
    print(f"CSV 文件保存成功")


def save_to_excel(df: pd.DataFrame, file_path: str) -> None:
    """
    保存 DataFrame 到 Excel 文件

    Args:
        df: 统计报告 DataFrame
        file_path: 输出文件路径
    """
    print(f"\n正在保存 Excel 文件: {file_path}")

    # 直接保存，不调整列宽（因为列数太多，超过165列）
    df.to_excel(file_path, index=False, sheet_name='统计报告', engine='openpyxl')

    print(f"Excel 文件保存成功")


def print_summary_statistics(df: pd.DataFrame, intervals: List[str],
                            interval_mapping: Dict[str, List[str]]) -> None:
    """
    打印统计摘要信息

    Args:
        df: 统计报告 DataFrame
        intervals: 相似度区间列表
        interval_mapping: 区间映射关系字典
    """
    print("\n" + "=" * 80)
    print("统计摘要信息")
    print("=" * 80)

    print(f"\n产业链环节总数: {len(df)}")
    print(f"标准化后区间总数: {len(intervals)}")
    print(f"标准区间宽度: {INTERVAL_WIDTH}")

    # 显示区间范围
    print(f"\n相似度区间范围:")
    print(f"  最小区间: {intervals[0]}")
    print(f"  最大区间: {intervals[-1]}")

    # 统计每个区间的总样本数
    print(f"\n各区间总样本数统计:")
    for interval in intervals[:10]:  # 只显示前10个区间
        sample_col = f'{interval}{THRESHOLD_SAMPLE_COUNT_SUFFIX}'
        total_samples = df[sample_col].sum()
        non_null_count = df[sample_col].notna().sum()
        print(f"  {interval}: {int(total_samples)} 个样本 (覆盖 {non_null_count} 个环节)")

    if len(intervals) > 10:
        print(f"  ... (还有 {len(intervals) - 10} 个区间)")

    # 统计整体符合率
    print(f"\n整体统计:")
    total_samples = 0
    total_matches = 0

    for interval in intervals:
        rate_col = f'{interval}{THRESHOLD_COMPLIANCE_RATE_SUFFIX}'
        sample_col = f'{interval}{THRESHOLD_SAMPLE_COUNT_SUFFIX}'

        for _, row in df.iterrows():
            if pd.notna(row[sample_col]):
                samples = int(row[sample_col])
                rate = row[rate_col]
                total_samples += samples
                total_matches += int(samples * rate)

    overall_match_rate = total_matches / total_samples if total_samples > 0 else 0
    print(f"  总样本数: {total_samples}")
    print(f"  总符合数: {total_matches}")
    print(f"  整体符合率: {overall_match_rate:.2%}")

    print("\n" + "=" * 80)


def main() -> None:
    """主函数"""
    print("=" * 80)
    print("产业链匹配统计报告生成工具")
    print("=" * 80)

    # 检查输入文件是否存在
    if not Path(INPUT_FILE_PATH).exists():
        print(f"错误: 输入文件不存在: {INPUT_FILE_PATH}")
        return

    # 加载数据
    data = load_json_data(INPUT_FILE_PATH)

    # 标准化数据（将非标准区间映射到标准区间）
    normalized_data, interval_mapping = normalize_data(data, INTERVAL_WIDTH)

    # 提取所有标准化后的相似度区间
    intervals = extract_all_intervals(normalized_data)

    # 生成统计表格
    df = generate_statistics_table(normalized_data, intervals)

    # 保存到文件
    save_to_csv(df, OUTPUT_CSV_PATH)
    save_to_excel(df, OUTPUT_EXCEL_PATH)

    # 打印摘要统计
    print_summary_statistics(df, intervals, interval_mapping)

    print(f"\n处理完成！")
    print(f"输出文件:")
    print(f"  - CSV: {OUTPUT_CSV_PATH}")
    print(f"  - Excel: {OUTPUT_EXCEL_PATH}")


if __name__ == "__main__":
    main()

