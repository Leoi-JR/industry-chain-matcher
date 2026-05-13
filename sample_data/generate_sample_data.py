#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate sample data for industry-chain-matcher.

Run this script once before executing the chain_matching_sop pipeline:

    python sample_data/generate_sample_data.py

What it creates
---------------
sample_data/
    chain_embeddings.npz          – embeddings for all 13 chain nodes (m_total=13, d=64)
    l0_embedding.npz              – embedding for the L0 root node (1, d=64)
    chain_type_classification.csv – A/B type label for each chain node
    chain_definitions.json        – text definitions for each node (used by LLM calibration)
    source_embeddings/
        source_part_1.npz         – source embeddings batch 1 (100 records, d=64)
        source_part_2.npz         – source embeddings batch 2 (100 records, d=64)
    source_texts/
        source_part_1.parquet     – source text batch 1 (id, source_text)
        source_part_2.parquet     – source text batch 2 (id, source_text)

chain_matching_sop/input_data/
    threshold_l0.json             – L0 similarity threshold
    threshold_spec.csv            – per-chain thresholds for specific chains
    threshold_other.csv           – per-chain thresholds for "other" chains

All embeddings use random normalised vectors (seed=42, d=64).
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent          # sample_data/
PROJECT_ROOT = SCRIPT_DIR.parent            # project root
INPUT_DATA_DIR = (
    PROJECT_ROOT / "chain_matching_sop" / "input_data"
)

# ---------------------------------------------------------------------------
# Industry chain structure
# ---------------------------------------------------------------------------
# Each tuple: (chain_name, type)
# type "A" = industry-specific application (requires L0 context)
# type "B" = general enabler (no L0 constraint)

L0_NAME = "示例产业链"

CHAIN_NODES = [
    # ── 核心零部件 ────────────────────────────────────────────────────────
    ("示例产业链——核心零部件——传感器——视觉传感器", "A"),
    ("示例产业链——核心零部件——传感器——力觉传感器", "A"),
    ("示例产业链——核心零部件——传感器——其他",      "A"),
    ("示例产业链——核心零部件——驱动系统——伺服电机", "B"),
    ("示例产业链——核心零部件——驱动系统——减速器",  "B"),
    ("示例产业链——核心零部件——驱动系统——其他",    "B"),
    # ── 软件系统 ────────────────────────────────────────────────────────
    ("示例产业链——软件系统——操作系统",   "A"),
    ("示例产业链——软件系统——控制软件",   "A"),
    ("示例产业链——软件系统——其他",       "A"),
    # ── 系统集成 ────────────────────────────────────────────────────────
    ("示例产业链——系统集成——产品制造",   "A"),
    ("示例产业链——系统集成——其他",       "A"),
    # ── 通用材料（B组，用于测试无L0约束逻辑） ────────────────────────
    ("示例产业链——通用材料——金属材料",   "B"),
    ("示例产业链——通用材料——其他",       "B"),
]

# Text definitions for each node (used in LLM threshold calibration)
CHAIN_DEFINITIONS = {
    L0_NAME: (
        "示例产业链是一条涵盖核心零部件、软件系统、系统集成及通用材料的"
        "综合性工业产业链，旨在支撑智能设备的研发、制造与应用。"
    ),
    "示例产业链——核心零部件——传感器——视觉传感器": (
        "视觉传感器是利用图像采集技术感知外部环境的元器件，包括工业相机、"
        "图像传感器芯片及相关光学组件，广泛用于机器视觉和自动化检测场景。"
    ),
    "示例产业链——核心零部件——传感器——力觉传感器": (
        "力觉传感器用于检测物体间的接触力、扭矩和压力信号，包括六维力传感器、"
        "触觉阵列传感器等，是机器人精密操作的核心感知部件。"
    ),
    "示例产业链——核心零部件——传感器——其他": (
        "其他类型传感器指除视觉传感器、力觉传感器之外的各类环境感知元器件，"
        "如温湿度传感器、气体传感器、激光雷达等。"
    ),
    "示例产业链——核心零部件——驱动系统——伺服电机": (
        "伺服电机是能够精确控制转速和位置的电动机，通过伺服驱动器实现高精度"
        "运动控制，是工业机器人、数控机床等精密运动设备的核心执行部件。"
    ),
    "示例产业链——核心零部件——驱动系统——减速器": (
        "减速器通过齿轮传动降低输出轴转速并增大输出扭矩，包括谐波减速器、"
        "RV减速器等，是机器人关节传动的关键精密机械部件。"
    ),
    "示例产业链——核心零部件——驱动系统——其他": (
        "其他驱动系统部件指除伺服电机、减速器之外的运动执行元件，"
        "包括液压驱动器、气动执行器、直线电机等。"
    ),
    "示例产业链——软件系统——操作系统": (
        "工业及机器人操作系统是管理硬件资源、提供实时控制能力的基础软件平台，"
        "包括实时操作系统（RTOS）和机器人操作系统（ROS）等。"
    ),
    "示例产业链——软件系统——控制软件": (
        "控制软件负责运动规划、轨迹生成和实时控制逻辑，包括机器人示教软件、"
        "运动控制库和工艺控制软件，直接驱动硬件完成作业任务。"
    ),
    "示例产业链——软件系统——其他": (
        "其他软件系统指除操作系统、控制软件之外的配套软件，包括仿真软件、"
        "视觉处理软件、工业互联网平台等。"
    ),
    "示例产业链——系统集成——产品制造": (
        "产品制造是将各类零部件和软件系统集成为完整可交付智能设备的环节，"
        "涵盖机械装配、电气集成、软件调试和出厂测试全流程。"
    ),
    "示例产业链——系统集成——其他": (
        "其他系统集成服务指产品制造之外的集成业务，如产线集成、系统方案设计、"
        "售后维护及改造升级服务。"
    ),
    "示例产业链——通用材料——金属材料": (
        "通用金属材料包括铝合金、钛合金、不锈钢等结构材料，广泛用于机械零部件"
        "加工，属于通用工业原材料，不依赖特定应用场景。"
    ),
    "示例产业链——通用材料——其他": (
        "其他通用材料指金属材料之外的工业原材料，包括工程塑料、碳纤维复合材料、"
        "导热绝缘材料等，具有广泛的跨行业适用性。"
    ),
}

# ---------------------------------------------------------------------------
# Sample source texts (generic business descriptions)
# ---------------------------------------------------------------------------
SOURCE_TEXT_TEMPLATES = [
    "研发、生产、销售{product}；提供{service}服务。",
    "从事{product}的设计与制造；{service}技术开发。",
    "{product}研发制造；{service}系统集成与技术服务。",
    "专注于{product}核心技术研究；提供{service}解决方案。",
    "{product}生产销售；{service}工程实施与运维服务。",
]

PRODUCTS = [
    "工业机器人及其控制系统",
    "伺服电机与驱动器",
    "机器视觉检测设备",
    "谐波减速器",
    "工业传感器模组",
    "数控机床",
    "铝合金精密结构件",
    "嵌入式实时操作系统",
    "机器人示教编程软件",
    "自动化产线集成系统",
    "碳纤维复合材料制品",
    "液压驱动执行机构",
    "工业互联网平台",
    "激光雷达传感器",
    "六维力矩传感器",
    "智能仓储物流设备",
    "精密注塑零部件",
    "电子元器件及模组",
    "工业计算机及控制板卡",
    "协作机器人整机",
]

SERVICES = [
    "自动化系统设计与集成",
    "智能制造整体解决方案",
    "产品检测与质量控制",
    "工业软件定制开发",
    "机器人应用培训与咨询",
    "精密加工与表面处理",
    "产线改造升级",
    "售后维保与远程运维",
    "工艺流程优化",
    "数字化工厂规划",
]


def _generate_source_texts(n: int, seed: int = 42) -> list[str]:
    """Generate n fake source texts."""
    rng = np.random.default_rng(seed)
    texts = []
    for i in range(n):
        template = SOURCE_TEXT_TEMPLATES[i % len(SOURCE_TEXT_TEMPLATES)]
        product = PRODUCTS[int(rng.integers(0, len(PRODUCTS)))]
        service = SERVICES[int(rng.integers(0, len(SERVICES)))]
        texts.append(template.format(product=product, service=service))
    return texts


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

def _random_unit_vectors(n: int, d: int, seed: int) -> np.ndarray:
    """Return (n, d) matrix of L2-normalised random float32 vectors."""
    rng = np.random.default_rng(seed)
    vecs = rng.standard_normal((n, d)).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.where(norms == 0, 1.0, norms)


# ---------------------------------------------------------------------------
# Main generation logic
# ---------------------------------------------------------------------------

def generate(d: int = 64, n_sources_per_part: int = 100) -> None:
    print("=" * 60)
    print("Generating sample data for industry-chain-matcher")
    print(f"  Embedding dimension : d = {d}")
    print(f"  Sources per file    : {n_sources_per_part}")
    print("=" * 60)

    SCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    (SCRIPT_DIR / "source_embeddings").mkdir(exist_ok=True)
    (SCRIPT_DIR / "source_texts").mkdir(exist_ok=True)
    INPUT_DATA_DIR.mkdir(parents=True, exist_ok=True)

    chain_names = np.array([name for name, _ in CHAIN_NODES])
    chain_types = [t for _, t in CHAIN_NODES]
    m_total = len(chain_names)

    # ── 1. Chain embeddings (m_total, d) ─────────────────────────────────
    chain_embeds = _random_unit_vectors(m_total, d, seed=0)
    np.savez_compressed(
        SCRIPT_DIR / "chain_embeddings.npz",
        chain_names=chain_names,
        embeddings=chain_embeds,
    )
    print(f"[1/7] chain_embeddings.npz  ({m_total} chains, d={d})")

    # ── 2. L0 embedding (1, d) ────────────────────────────────────────────
    l0_embed = _random_unit_vectors(1, d, seed=1)
    np.savez_compressed(
        SCRIPT_DIR / "l0_embedding.npz",
        chain_names=np.array([L0_NAME]),
        embeddings=l0_embed,
    )
    print(f"[2/7] l0_embedding.npz  (L0='{L0_NAME}', d={d})")

    # ── 3. Chain type classification CSV ──────────────────────────────────
    df_type = pd.DataFrame({"chain_name": chain_names, "type": chain_types})
    df_type.to_csv(SCRIPT_DIR / "chain_type_classification.csv", index=False)
    n_a = sum(1 for t in chain_types if t == "A")
    n_b = sum(1 for t in chain_types if t == "B")
    print(f"[3/7] chain_type_classification.csv  (A={n_a}, B={n_b})")

    # ── 4. Chain definitions JSON ─────────────────────────────────────────
    definitions_list = []
    for name, _ in [(L0_NAME, "A")] + CHAIN_NODES:
        definitions_list.append({
            "name": name,
            "definition": CHAIN_DEFINITIONS.get(name, f"{name}的示例定义文本。"),
        })
    with open(SCRIPT_DIR / "chain_definitions.json", "w", encoding="utf-8") as f:
        json.dump(definitions_list, f, ensure_ascii=False, indent=2)
    print(f"[4/7] chain_definitions.json  ({len(definitions_list)} entries)")

    # ── 5. Source embeddings + text parquet (2 batches) ───────────────────
    texts = _generate_source_texts(n_sources_per_part * 2)
    for part in range(1, 3):
        start_id = (part - 1) * n_sources_per_part + 1
        ids = np.arange(start_id, start_id + n_sources_per_part, dtype=np.int64)
        batch_texts = texts[(part - 1) * n_sources_per_part: part * n_sources_per_part]

        embeds = _random_unit_vectors(n_sources_per_part, d, seed=100 + part)
        np.savez_compressed(
            SCRIPT_DIR / "source_embeddings" / f"source_part_{part}.npz",
            ids=ids,
            embeddings=embeds,
        )

        df_text = pd.DataFrame({"id": ids, "source_text": batch_texts})
        df_text.to_parquet(
            SCRIPT_DIR / "source_texts" / f"source_part_{part}.parquet",
            index=False,
        )
        print(
            f"[5/7] source_part_{part}: embeddings + texts  "
            f"(ids {ids[0]}–{ids[-1]})"
        )

    # ── 6. Threshold files for chain_matching_sop/input_data/ ─────────────
    spec_names  = [name for name, _ in CHAIN_NODES if not name.endswith("其他")]
    other_names = [name for name, _ in CHAIN_NODES if name.endswith("其他")]

    # Random unit vectors in d=64 have E[cos_sim] ≈ 0, std ≈ 0.125.
    # Use low thresholds so the demo pipeline produces visible matches.
    with open(INPUT_DATA_DIR / "threshold_l0.json", "w", encoding="utf-8") as f:
        json.dump({"T_L0": 0.05}, f, indent=2)

    pd.DataFrame({
        "chain_name": spec_names,
        "threshold":  [0.10] * len(spec_names),
    }).to_csv(INPUT_DATA_DIR / "threshold_spec.csv", index=False)

    pd.DataFrame({
        "chain_name": other_names,
        "threshold":  [0.08] * len(other_names),
    }).to_csv(INPUT_DATA_DIR / "threshold_other.csv", index=False)

    print(
        f"[6/7] threshold files  "
        f"(L0=0.05, {len(spec_names)} spec=0.10, {len(other_names)} other=0.08)"
    )

    # ── 7. Summary ────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("Sample data generated successfully.")
    print()
    print("Next steps:")
    print("  python chain_matching_sop/prepare_chain_config.py")
    print("  python chain_matching_sop/run_similarity.py")
    print("  python chain_matching_sop/apply_matching.py")
    print("  python chain_matching_sop/export_results.py")
    print("=" * 60)


if __name__ == "__main__":
    generate()
