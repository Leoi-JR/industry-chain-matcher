#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
阈值生成脚本

功能：
1. 读取 chain_matching_statistics.xlsx 文件
2. 根据符合率阈值计算每个产业链环节的相似度阈值
3. 将结果分组保存为三个配置文件：
   - threshold_l0.json: L0 层级阈值（产业链环节名称不包含"——"）
   - threshold_other.csv: "其他"类型阈值（产业链环节名称以"其他"结尾）
   - threshold_spec.csv: 特定环节阈值（产业链环节名称包含"——"但不以"其他"结尾）

工作流位置：
- 位于 calibrate_thresholds_llm.py 和 apply_matching.py 之间
- 输入：similarity_output/chain_matching_statistics.xlsx
- 输出：input_data/threshold_l0.json, threshold_other.csv, threshold_spec.csv
"""

import sys
import json
from pathlib import Path
from typing import Dict, Tuple, List
import pandas as pd
import numpy as np

# 从配置文件导入参数
from config.data_config import (
    # 阈值计算参数
    THRESHOLD_COMPLIANCE_RATE,
    THRESHOLD_DEFAULT_VALUE,
    THRESHOLD_L0_DEFAULT_VALUE,
    # 列名配置
    THRESHOLD_CHAIN_NAME_COLUMN,
    THRESHOLD_COMPLIANCE_RATE_SUFFIX,
    # 文件路径配置
    THRESHOLD_INPUT_FILE_PATH,
    THRESHOLD_OUTPUT_DIR,
    THRESHOLD_L0_PATH,
    THRESHOLD_OTHER_PATH,
    THRESHOLD_SPEC_PATH,
)


# ============================================================================
# 数据加载函数
# ============================================================================

def load_statistics_file(file_path: Path) -> pd.DataFrame:
    """
    加载产业链匹配统计文件
    
    Args:
        file_path: Excel 文件路径
        
    Returns:
        统计数据 DataFrame
        
    Raises:
        FileNotFoundError: 如果文件不存在
        ValueError: 如果文件格式不正确
    """
    print("=" * 80)
    print("加载统计数据文件")
    print("=" * 80)
    print(f"文件路径: {file_path}")
    
    if not file_path.exists():
        raise FileNotFoundError(f"统计文件不存在: {file_path}")
    
    try:
        df = pd.read_excel(file_path)
        print(f"成功加载数据: {len(df)} 行, {len(df.columns)} 列")
        
        # 验证必需的列是否存在
        if THRESHOLD_CHAIN_NAME_COLUMN not in df.columns:
            raise ValueError(f"缺少必需的列: {THRESHOLD_CHAIN_NAME_COLUMN}")
        
        print(f"产业链环节数量: {len(df)}")
        print()
        
        return df
        
    except Exception as e:
        raise ValueError(f"读取文件失败: {str(e)}")


# ============================================================================
# 区间解析函数
# ============================================================================

def parse_interval_from_column(column_name: str) -> Tuple[float, float]:
    """
    从列名中解析区间范围
    
    Args:
        column_name: 列名，格式如 "[0.400, 0.425)_符合率"
        
    Returns:
        (下界, 上界) 元组
        
    Raises:
        ValueError: 如果列名格式不正确
    """
    try:
        # 提取区间部分，例如 "[0.400, 0.425)"
        interval_str = column_name.split('_')[0]
        
        # 移除方括号和圆括号
        interval_str = interval_str.strip('[]() ')
        
        # 分割并转换为浮点数
        parts = interval_str.split(',')
        lower_bound = float(parts[0].strip())
        upper_bound = float(parts[1].strip())
        
        return lower_bound, upper_bound
        
    except Exception as e:
        raise ValueError(f"无法解析列名 '{column_name}': {str(e)}")


def extract_intervals_from_dataframe(df: pd.DataFrame) -> List[Tuple[str, float, float]]:
    """
    从 DataFrame 中提取所有相似度区间信息
    
    Args:
        df: 统计数据 DataFrame
        
    Returns:
        区间信息列表，每个元素为 (区间字符串, 下界, 上界) 元组，按下界排序
    """
    intervals = []
    
    for column in df.columns:
        if column.endswith(THRESHOLD_COMPLIANCE_RATE_SUFFIX):
            # 提取区间字符串（去掉后缀）
            interval_str = column[:-len(THRESHOLD_COMPLIANCE_RATE_SUFFIX)]
            
            # 解析区间范围
            lower_bound, upper_bound = parse_interval_from_column(column)
            
            intervals.append((interval_str, lower_bound, upper_bound))
    
    # 按下界排序
    intervals.sort(key=lambda x: x[1])
    
    print(f"提取到 {len(intervals)} 个相似度区间")
    print(f"区间范围: {intervals[0][0]} 到 {intervals[-1][0]}")
    print()
    
    return intervals


# ============================================================================
# 阈值计算函数
# ============================================================================

def calculate_threshold_for_chain(
    row: pd.Series,
    intervals: List[Tuple[str, float, float]],
    compliance_threshold: float
) -> float:
    """
    计算单个产业链环节的相似度阈值
    
    逻辑：
    1. 从最低相似度区间开始遍历
    2. 找到第一个符合率 >= compliance_threshold 的区间
    3. 返回该区间的下界作为阈值
    4. 如果所有区间都未达到要求，返回 THRESHOLD_DEFAULT_VALUE

    Args:
        row: DataFrame 的一行数据（代表一个产业链环节）
        intervals: 区间信息列表
        compliance_threshold: 符合率阈值

    Returns:
        计算得到的相似度阈值
    """
    for interval_str, lower_bound, upper_bound in intervals:
        # 构造符合率列名
        rate_column = f"{interval_str}{THRESHOLD_COMPLIANCE_RATE_SUFFIX}"

        # 获取符合率值
        compliance_rate = row[rate_column]

        # 跳过空值
        if pd.isna(compliance_rate):
            continue

        # 检查是否达到阈值
        if compliance_rate >= compliance_threshold:
            return lower_bound

    # 所有区间都未达到要求，返回默认阈值
    return THRESHOLD_DEFAULT_VALUE


def calculate_all_thresholds(
    df: pd.DataFrame,
    intervals: List[Tuple[str, float, float]],
    compliance_threshold: float
) -> Dict[str, float]:
    """
    计算所有产业链环节的相似度阈值
    
    Args:
        df: 统计数据 DataFrame
        intervals: 区间信息列表
        compliance_threshold: 符合率阈值
        
    Returns:
        字典，键为产业链环节名称，值为相似度阈值
    """
    print("=" * 80)
    print("计算相似度阈值")
    print("=" * 80)
    print(f"符合率阈值: {compliance_threshold}")
    print()
    
    thresholds = {}
    default_count = 0

    for _, row in df.iterrows():
        chain_name = row[THRESHOLD_CHAIN_NAME_COLUMN]
        threshold = calculate_threshold_for_chain(row, intervals, compliance_threshold)
        thresholds[chain_name] = threshold

        if threshold == THRESHOLD_DEFAULT_VALUE:
            default_count += 1
    
    print(f"计算完成:")
    print(f"  总环节数: {len(thresholds)}")
    print(f"  使用默认阈值的环节数: {default_count}")
    print()
    
    return thresholds


# ============================================================================
# 分组函数
# ============================================================================

def classify_chain_by_name(chain_name: str) -> str:
    """
    根据产业链环节名称判断其类型
    
    分类规则：
    - "l0": 不包含"——"（L0 层级）
    - "other": 以"其他"结尾
    - "spec": 包含"——"但不以"其他"结尾（特定环节）
    
    Args:
        chain_name: 产业链环节名称
        
    Returns:
        类型标识: "l0", "other", 或 "spec"
    """
    if "——" not in chain_name:
        return "l0"
    elif chain_name.endswith("其他"):
        return "other"
    else:
        return "spec"


def group_thresholds_by_type(thresholds: Dict[str, float]) -> Dict[str, Dict[str, float]]:
    """
    将阈值字典按类型分组
    
    Args:
        thresholds: 阈值字典
        
    Returns:
        分组后的字典，格式为 {类型: {产业链环节名称: 阈值}}
    """
    print("=" * 80)
    print("分组阈值数据")
    print("=" * 80)
    
    grouped = {
        "l0": {},
        "other": {},
        "spec": {}
    }
    
    for chain_name, threshold in thresholds.items():
        chain_type = classify_chain_by_name(chain_name)
        grouped[chain_type][chain_name] = threshold
    
    print(f"分组结果:")
    print(f"  L0 层级: {len(grouped['l0'])} 个环节")
    print(f"  其他类型: {len(grouped['other'])} 个环节")
    print(f"  特定环节: {len(grouped['spec'])} 个环节")
    print()
    
    return grouped


# ============================================================================
# 文件保存函数
# ============================================================================

def save_l0_threshold(thresholds: Dict[str, float], output_path: Path) -> None:
    """
    保存 L0 层级阈值到 JSON 文件。
    若 LLM 未覆盖 L0，则保留现有文件中的值；文件不存在时使用默认值。
    """
    print(f"保存 L0 阈值: {output_path}")

    if len(thresholds) == 0:
        # 无 LLM 数据：保留现有文件，否则使用默认值
        if output_path.exists():
            with open(output_path, encoding='utf-8') as f:
                existing = json.load(f)
            t_l0 = existing.get("T_L0", THRESHOLD_L0_DEFAULT_VALUE)
            print(f"  LLM 未覆盖 L0，保留现有值: {t_l0}")
        else:
            t_l0 = THRESHOLD_L0_DEFAULT_VALUE
            print(f"  LLM 未覆盖 L0，使用默认值: {t_l0}")
    elif len(thresholds) == 1:
        t_l0 = list(thresholds.values())[0]
        print(f"  L0 阈值: {t_l0}")
    else:
        t_l0 = np.mean(list(thresholds.values()))
        print(f"  警告: 发现 {len(thresholds)} 个 L0 环节，取平均值: {t_l0}")

    output_data = {"T_L0": float(t_l0)}
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"  ✓ 保存成功")
    print()


def save_csv_threshold(thresholds: Dict[str, float], output_path: Path, file_type: str) -> None:
    """
    保存阈值到 CSV 文件。
    以现有文件为基准，将 LLM 计算出的阈值合并进去（仅覆盖有新数据的行），
    LLM 未覆盖的链条保留原值；文件不存在时纯新建。
    """
    print(f"保存 {file_type} 阈值: {output_path}")

    if output_path.exists():
        existing_df = pd.read_csv(output_path)
        # 以现有 chain_name 为基准，用 LLM 结果覆盖对应行
        existing_df = existing_df.set_index("chain_name")
        for chain_name, threshold in thresholds.items():
            existing_df.loc[chain_name, "threshold"] = threshold
        df = existing_df.reset_index().rename(columns={"index": "chain_name"})
        updated = len(thresholds)
        print(f"  合并完成: {len(df)} 条记录，其中 {updated} 条由 LLM 更新，"
              f"{len(df) - updated} 条保留原值")
    else:
        if len(thresholds) == 0:
            df = pd.DataFrame(columns=["chain_name", "threshold"])
            print(f"  警告: 无现有文件且 LLM 无数据，创建空文件")
        else:
            df = pd.DataFrame([
                {"chain_name": k, "threshold": v} for k, v in thresholds.items()
            ])
            print(f"  新建文件: {len(df)} 条记录")

    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"  ✓ 保存成功")
    print()


def save_all_thresholds(grouped_thresholds: Dict[str, Dict[str, float]]) -> None:
    """
    保存所有分组后的阈值到文件
    
    Args:
        grouped_thresholds: 分组后的阈值字典
    """
    print("=" * 80)
    print("保存阈值配置文件")
    print("=" * 80)
    
    # 确保输出目录存在
    THRESHOLD_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"输出目录: {THRESHOLD_OUTPUT_DIR}")
    print()

    # 保存 L0 阈值
    save_l0_threshold(grouped_thresholds["l0"], THRESHOLD_L0_PATH)

    # 保存其他类型阈值
    save_csv_threshold(grouped_thresholds["other"], THRESHOLD_OTHER_PATH, "其他类型")

    # 保存特定环节阈值
    save_csv_threshold(grouped_thresholds["spec"], THRESHOLD_SPEC_PATH, "特定环节")


# ============================================================================
# 主函数
# ============================================================================

def main() -> None:
    """主函数"""
    print("\n" + "=" * 80)
    print("阈值生成脚本")
    print("=" * 80)
    print()
    
    try:
        # 1. 加载统计数据
        df = load_statistics_file(THRESHOLD_INPUT_FILE_PATH)

        # 2. 提取区间信息
        intervals = extract_intervals_from_dataframe(df)

        # 3. 计算阈值
        thresholds = calculate_all_thresholds(df, intervals, THRESHOLD_COMPLIANCE_RATE)

        # 4. 分组
        grouped_thresholds = group_thresholds_by_type(thresholds)

        # 5. 保存结果
        save_all_thresholds(grouped_thresholds)

        # 6. 完成
        print("=" * 80)
        print("处理完成！")
        print("=" * 80)
        print("输出文件:")
        print(f"  - {THRESHOLD_L0_PATH}")
        print(f"  - {THRESHOLD_OTHER_PATH}")
        print(f"  - {THRESHOLD_SPEC_PATH}")
        print()
        
    except Exception as e:
        print(f"\n错误: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()

