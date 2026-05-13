#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Result export module: process matching output and generate final matching results.
"""

import logging
import sys
from pathlib import Path
from typing import List, Any
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.io_utils import ensure_dir
from config.data_config import (
    CHAIN_EMBEDDINGS_PATH,
    CHAIN_EMBEDDINGS_KEYS,
    MATCHING_OUTPUT_DIR,
    RESULTS_SPLIT_INFO_DIR,
    RESULTS_DIR,
    RESULTS_OUTPUT_FILE_NAME,
    RESULTS_SOURCE_FILE_PATTERN,
    RESULTS_SOURCE_ID_FIELD,
    RESULTS_SOURCE_CONTENT_FIELDS,
    RESULTS_LOG_LEVEL
)


def setup_logging(log_level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def extract_matches_from_npz(npz_file_path: Path) -> List[List[Any]]:
    logging.info(f"正在处理 NPZ 文件: {npz_file_path}")

    try:
        npz_data = np.load(npz_file_path)
        result_matrix = npz_data['result_matrix']
        source_ids = npz_data['source_ids']

        logging.info(f"结果矩阵形状: {result_matrix.shape}")
        logging.info(f"源 ID 数量: {len(source_ids)}")

        nonzero_indices = np.where(result_matrix > 0)
        matches = []

        for i in range(len(nonzero_indices[0])):
            m = nonzero_indices[0][i]
            n = nonzero_indices[1][i]
            info_id = source_ids[n]
            similarity = result_matrix[m, n]
            matches.append([int(m), int(info_id), float(similarity)])

        logging.info(f"从 {npz_file_path.name} 提取到 {len(matches)} 个匹配")
        return matches

    except Exception as e:
        logging.error(f"处理 NPZ 文件 {npz_file_path} 时出错: {e}")
        raise


def process_all_npz_files(matching_output_dir: Path) -> pd.DataFrame:
    logging.info(f"开始处理匹配输出目录: {matching_output_dir}")

    npz_files = list(matching_output_dir.glob("result_matrix_batch_*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"在 {matching_output_dir} 中未找到 NPZ 文件")

    logging.info(f"找到 {len(npz_files)} 个 NPZ 文件")

    all_matches = []
    for npz_file in tqdm(npz_files, desc="处理 NPZ 文件"):
        matches = extract_matches_from_npz(npz_file)
        all_matches.extend(matches)

    logging.info(f"总共提取到 {len(all_matches)} 个匹配")

    if not all_matches:
        logging.warning("未找到任何匹配结果")
        return pd.DataFrame(columns=["chain_id", "info_id", "similarity"])

    df = pd.DataFrame(all_matches, columns=["chain_id", "info_id", "similarity"])
    df["chain_id"] = df["chain_id"].astype(int)
    df["info_id"] = df["info_id"].astype(int)
    df["similarity"] = df["similarity"].astype(float)

    logging.info(f"构建的匹配结果 DataFrame 形状: {df.shape}")
    logging.info(f"唯一 chain_id 数量: {df['chain_id'].nunique()}")
    logging.info(f"唯一 info_id 数量: {df['info_id'].nunique()}")

    return df


def process_source_files(
    split_info_dir: Path,
    target_info_ids: set,
    source_id_field: str = "id",
    source_content_fields: List[str] = None,
    file_pattern: str = "*.parquet"
) -> pd.DataFrame:
    if source_content_fields is None:
        source_content_fields = ["source_text"]

    logging.info(f"开始处理源数据目录: {split_info_dir}")
    logging.info(f"目标 info_id 数量: {len(target_info_ids)}")
    logging.info(f"源ID字段: {source_id_field}")
    logging.info(f"内容字段: {source_content_fields}")

    source_files = list(split_info_dir.glob(file_pattern))
    if not source_files:
        raise FileNotFoundError(f"在 {split_info_dir} 中未找到匹配模式 '{file_pattern}' 的文件")

    logging.info(f"找到 {len(source_files)} 个源数据文件")

    filtered_dfs = []
    total_processed = 0
    total_matched = 0

    for source_file in tqdm(source_files, desc="处理源数据文件"):
        try:
            df = pd.read_parquet(source_file)
            total_processed += len(df)

            if source_id_field not in df.columns:
                raise ValueError(f"文件 {source_file.name} 中不存在字段 '{source_id_field}'")

            missing_fields = [field for field in source_content_fields if field not in df.columns]
            if missing_fields:
                raise ValueError(f"文件 {source_file.name} 中不存在字段: {missing_fields}")

            df[source_id_field] = df[source_id_field].astype(int)
            filtered_df = df[df[source_id_field].isin(target_info_ids)]

            if not filtered_df.empty:
                columns_to_keep = [source_id_field] + source_content_fields
                filtered_df = filtered_df[columns_to_keep]
                filtered_dfs.append(filtered_df)
                total_matched += len(filtered_df)
                logging.debug(f"{source_file.name}: {len(filtered_df)} 条匹配记录")

        except Exception as e:
            logging.error(f"处理源数据文件 {source_file} 时出错: {e}")
            raise

    logging.info(f"总共处理了 {total_processed} 条记录，匹配到 {total_matched} 条记录")

    if not filtered_dfs:
        logging.warning("未找到任何匹配的源数据记录")
        columns_to_return = ["info_id"] + source_content_fields
        return pd.DataFrame(columns=columns_to_return)

    combined_df = pd.concat(filtered_dfs, ignore_index=True)

    original_count = len(combined_df)
    combined_df = combined_df.drop_duplicates(subset=[source_id_field])
    final_count = len(combined_df)

    if original_count != final_count:
        logging.info(f"去重后从 {original_count} 条记录减少到 {final_count} 条记录")

    combined_df = combined_df.rename(columns={source_id_field: "info_id"})
    logging.info(f"最终源数据 DataFrame 形状: {combined_df.shape}")

    return combined_df


def load_and_validate_chain_names(matches_df: pd.DataFrame) -> np.ndarray:
    logging.info("开始加载产业链名称")

    if not CHAIN_EMBEDDINGS_PATH.exists():
        raise FileNotFoundError(f"产业链嵌入文件不存在: {CHAIN_EMBEDDINGS_PATH}")

    logging.info(f"从文件加载产业链名称: {CHAIN_EMBEDDINGS_PATH}")
    data = np.load(CHAIN_EMBEDDINGS_PATH)

    chain_names_key = CHAIN_EMBEDDINGS_KEYS['chain_names']
    if chain_names_key not in data:
        raise KeyError(f"产业链嵌入文件中未找到 '{chain_names_key}' 键")

    chain_names = data[chain_names_key]
    logging.info(f"成功加载 {len(chain_names)} 个产业链名称")

    unique_chain_ids = matches_df['chain_id'].unique()
    unique_chain_count = len(unique_chain_ids)
    chain_names_count = len(chain_names)

    logging.info(f"匹配结果中唯一 chain_id 数量: {unique_chain_count}")
    logging.info(f"产业链名称数组长度: {chain_names_count}")

    max_chain_id = max(unique_chain_ids)
    min_chain_id = min(unique_chain_ids)

    if min_chain_id < 0 or max_chain_id >= chain_names_count:
        raise ValueError(
            f"chain_id 超出有效范围。范围: {min_chain_id}-{max_chain_id}, "
            f"但产业链名称数组长度为: {chain_names_count}"
        )

    logging.info("产业链名称数据验证通过")
    return chain_names


def merge_and_save_results(
    matches_df: pd.DataFrame,
    source_data_df: pd.DataFrame,
    final_output_dir: Path,
    output_file_name: str,
    source_content_fields: List[str] = None
) -> None:
    if source_content_fields is None:
        source_content_fields = ["source_text"]

    logging.info("开始合并数据")
    ensure_dir(final_output_dir)

    final_df = matches_df.merge(
        source_data_df,
        left_on='info_id',
        right_on='info_id',
        how='left'
    )

    logging.info(f"合并后的 DataFrame 形状: {final_df.shape}")

    for field in source_content_fields:
        missing_count = final_df[field].isna().sum()
        if missing_count > 0:
            logging.warning(f"有 {missing_count} 条记录缺少 '{field}' 数据")

    logging.info("步骤 3.1: 添加产业链名称")
    chain_names = load_and_validate_chain_names(matches_df)
    final_df['chain_name'] = final_df['chain_id'].apply(lambda x: chain_names[x])

    output_columns = ['chain_id', 'chain_name', 'info_id', 'similarity'] + source_content_fields
    final_df = final_df[output_columns]

    logging.info("成功添加产业链名称列")
    logging.info(f"最终 DataFrame 形状: {final_df.shape}")
    logging.info(f"最终 DataFrame 列: {list(final_df.columns)}")

    output_file = final_output_dir / output_file_name
    final_df.to_parquet(output_file, index=False)

    logging.info(f"最终结果已保存到: {output_file}")
    logging.info(f"最终结果统计:")
    logging.info(f"  总记录数: {len(final_df)}")
    logging.info(f"  唯一 chain_id 数量: {final_df['chain_id'].nunique()}")
    logging.info(f"  唯一 info_id 数量: {final_df['info_id'].nunique()}")
    logging.info(f"  相似度范围: {final_df['similarity'].min():.4f} - {final_df['similarity'].max():.4f}")
    logging.info(f"  平均相似度: {final_df['similarity'].mean():.4f}")


def export_final_results() -> None:
    """Main entry point for result export."""
    setup_logging(RESULTS_LOG_LEVEL)

    matching_output_dir = MATCHING_OUTPUT_DIR
    split_info_dir = RESULTS_SPLIT_INFO_DIR
    final_output_dir = RESULTS_DIR
    source_file_pattern = RESULTS_SOURCE_FILE_PATTERN
    source_id_field = RESULTS_SOURCE_ID_FIELD
    source_content_fields = RESULTS_SOURCE_CONTENT_FIELDS
    output_file_name = RESULTS_OUTPUT_FILE_NAME

    if not matching_output_dir.exists():
        logging.error(f"匹配输出目录不存在: {matching_output_dir}")
        sys.exit(1)

    if not split_info_dir.exists():
        logging.error(f"源数据目录不存在: {split_info_dir}")
        sys.exit(1)

    try:
        logging.info("=== 开始导出结果 ===")

        logging.info("步骤 1: 处理 NPZ 文件")
        matches_df = process_all_npz_files(matching_output_dir)

        if matches_df.empty:
            logging.error("未找到任何匹配结果，退出")
            sys.exit(1)

        target_info_ids = set(matches_df['info_id'].unique())

        logging.info("步骤 2: 处理源数据文件")
        source_data_df = process_source_files(
            split_info_dir,
            target_info_ids,
            source_id_field=source_id_field,
            source_content_fields=source_content_fields,
            file_pattern=source_file_pattern
        )

        logging.info("步骤 3: 合并数据、添加产业链名称并保存最终结果")
        merge_and_save_results(
            matches_df,
            source_data_df,
            final_output_dir,
            output_file_name,
            source_content_fields=source_content_fields
        )

        logging.info("=== 结果导出完成 ===")

    except Exception as e:
        logging.error(f"处理过程中出现错误: {e}")
        sys.exit(1)
