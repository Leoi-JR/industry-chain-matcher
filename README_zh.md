# 产业链挂链系统

基于矩阵运算的源文本自动匹配系统，将任意源文本批量匹配到预定义产业链图谱的具体环节。

[Algorithm Specification (English README)](README.md) | [**算法可视化说明**](https://leoi-jr.github.io/industry-chain-matcher/index.html)

---

## 概述

给定一个包含 `m` 个环节的产业链图谱和 `n` 条源文本文档，本系统通过向量相似度和一套可配置的匹配规则，高效地将每条文档映射到最相关的产业链环节。

**核心特性：**
- 批量矩阵运算（无逐条循环），可扩展至百万级文档
- 双类型链路逻辑：特定型链路（A组）需通过 L0 根节点相似度门控，通用型链路（B组）不需要
- 排他逻辑："其他"兜底节点仅在同级具体节点均未匹配时才命中
- LLM 辅助的阈值标定流水线
- 可选 GPU 加速（基于 CuPy）

---

## 算法简述

每条文档的 embedding 向量 `d` 与三组链路 embedding 分别计算相似度：

```
Sim_L0           = L0_Embed           @ D.T   →  (1,       n)
Sim_Spec         = Spec_Embeds        @ D.T   →  (m_spec,  n)
Sim_Other_Parent = OtherParent_Embeds @ D.T   →  (m_other, n)
```

最终匹配掩码的计算规则：

| 链路类型 | 匹配条件 |
|---------|---------|
| 具体链路（A 组） | `相似度 > 阈值` **且** `Sim_L0 > T_L0` |
| 具体链路（B 组） | `相似度 > 阈值`（无 L0 门控） |
| "其他"兜底节点 | 父级相似度 > 阈值 **且** 满足类型门控 **且** 同级具体节点均未匹配 |

详细算法说明见 [算法可视化页面](https://leoi-jr.github.io/industry-chain-matcher/index.html)。

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 生成示例数据

```bash
python sample_data/generate_sample_data.py
```

该脚本会生成一条包含 13 个节点的虚构产业链和 200 条模拟源文本文档，以及配套的阈值文件——足以跑通完整流水线。

### 3. 运行匹配流水线

```bash
# 预处理配置矩阵（每次更新链路定义后运行一次）
python chain_matching_sop/prepare_chain_config.py

# 第一步：计算所有源文本批次的原始相似度
python chain_matching_sop/run_similarity.py

# 第二步：应用阈值和匹配逻辑
python chain_matching_sop/apply_matching.py

# 第三步：导出最终结果为 Parquet 格式
python chain_matching_sop/export_results.py
```

### 4. 查看结果

```python
import pandas as pd
df = pd.read_parquet("chain_matching_sop/results/final_matching_results.parquet")
print(df.columns.tolist())
# ['chain_id', 'chain_name', 'info_id', 'similarity', 'source_text']
print(df.head())
```

---

## 目录结构

```
industry-chain-matcher/
│
├── sample_data/                    # 示例数据（可提交到版本库）
│   ├── generate_sample_data.py     # 一键生成脚本，首次使用先运行此脚本
│   ├── chain_embeddings.npz        # 链路节点 embedding（生成后）
│   ├── l0_embedding.npz            # L0 根节点 embedding（生成后）
│   ├── chain_type_classification.csv
│   ├── chain_definitions.json      # 链路文本定义，供 LLM 标定使用
│   ├── source_embeddings/          # 源文本 embedding（生成后）
│   └── source_texts/               # 源文本原文（生成后）
│
├── chain_matching_sop/             # 核心匹配算法
│   ├── config/
│   │   ├── data_config.py          # 所有文件路径与格式规范
│   │   └── sop_config.py           # 算法参数
│   ├── data_preparation/           # prepare_chain_config.py 子模块
│   ├── similarity/                 # run_similarity.py 子模块
│   ├── matching/                   # apply_matching.py 子模块
│   ├── result_export/              # export_results.py 子模块
│   ├── utils/                      # I/O 与矩阵运算工具
│   ├── prompt/                     # LLM 提示词模板
│   ├── input_data/                 # 阈值配置文件（见下文）
│   │   ├── threshold_l0.json
│   │   ├── threshold_spec.csv
│   │   └── threshold_other.csv
│   ├── prepare_chain_config.py     # 步骤 0：构建配置矩阵
│   ├── run_similarity.py           # 步骤 1：计算相似度
│   ├── sample_for_calibration.py   # 步骤 1.5a：按相似度分箱采样
│   ├── calibrate_thresholds_llm.py # 步骤 1.5b：LLM 阈值标定
│   ├── compute_compliance_stats.py # 步骤 1.5c：汇总符合率统计
│   ├── export_thresholds.py        # 步骤 1.5d：生成阈值文件
│   ├── apply_matching.py           # 步骤 2：应用匹配逻辑
│   └── export_results.py           # 步骤 3：导出结果
│
├── embedding_code/                 # 上游流水线：链路定义与向量化
│   ├── llm_define_chains.py              # 步骤 A1：LLM 生成链路文本定义
│   ├── parse_chain_definitions.py        # 步骤 A2：格式化与结构化定义
│   ├── embed_chain_definitions.py        # 步骤 A3：链路定义向量化
│   ├── preprocess_source_texts.py        # 步骤 B1：源文本预处理
│   ├── embed_source_texts.py             # 步骤 B2：源文本向量化
│   ├── embed_utils.py                    # 共享 embedding 工具函数
│   └── config.yaml                       # embedding 脚本配置
│
├── .env.example                    # 环境变量模板
├── requirements.txt
└── README.md
```

---

## 使用自己的数据

要在真实产业链上运行，需要准备以下两类输入：

> **硬件要求**：向量化流水线（`embedding_code/`）需要支持 CUDA 的 GPU，
> 使用 `flash_attention_2` + `float16`，CPU 环境下无法运行。
> 核心匹配流水线（`chain_matching_sop/`）无 GPU 要求。

### 步骤 A — 准备链路 embedding（上游流水线）

1. 将产业链 Excel 文件放置于 `sample_data/sample_industry_chain.xlsx`
2. 配置 `embedding_code/config.yaml`
3. 设置 embedding 模型路径：
   ```bash
   export EMBEDDING_MODEL_PATH=/path/to/your/embedding-model
   ```
4. 按顺序运行：
   ```bash
   python embedding_code/llm_define_chains.py        # 生成链路文本定义
   python embedding_code/parse_chain_definitions.py  # 格式化定义
   python embedding_code/embed_chain_definitions.py  # 链路定义向量化
   ```

5. 修改 `chain_matching_sop/config/data_config.py`，将 `CHAIN_EMBEDDINGS_PATH`、`L0_EMBEDDINGS_PATH`、`CHAIN_TYPE_CLASSIFICATION_PATH` 指向生成的文件。

### 步骤 B — 准备源文本 embedding

```bash
python embedding_code/preprocess_source_texts.py   # 清洗与分批源文本

# 向量化源文本（默认列名）
python embedding_code/embed_source_texts.py --input_file source_texts_split/batch_0.parquet

# 向量化其他文本类型
python embedding_code/embed_source_texts.py \
  --input_file patents/batch_0.parquet \
  --text_column custom_text_field \
  --id_column patent_id \
  --output_dir patent_embeddings
```

> **文件命名约定**：`embed_source_texts.py` 根据输入文件名自动派生输出文件名（如 `source_part_0.parquet` → `source_part_0_embeddings.npz`）。下游 `data_config.py` 通过 `SOURCE_EMBEDDINGS_PATTERN = "source_part_*.npz"` 匹配这些文件，因此输入文件必须以 `source_part_` 为前缀。`preprocess_source_texts.py` 默认使用该前缀。如需自定义文件名，请同步修改 `data_config.py` 中的 `SOURCE_EMBEDDINGS_PATTERN`。

修改 `data_config.py` 中的 `SOURCE_EMBEDDINGS_DIR`，指向你的 embedding 输出目录。

### 步骤 C — 标定阈值（可选，但推荐）

运行步骤 1 后，使用 LLM 流水线确定每个链路的相似度阈值：

```bash
python chain_matching_sop/sample_for_calibration.py        # 按相似度分箱采样
python chain_matching_sop/calibrate_thresholds_llm.py      # LLM 判断
python chain_matching_sop/compute_compliance_stats.py      # 计算符合率
python chain_matching_sop/export_thresholds.py             # 生成阈值文件
```

也可以手动编辑 `chain_matching_sop/input_data/threshold_spec.csv`、`threshold_other.csv` 和 `threshold_l0.json`。

### 数据格式参考

| 文件 | 必需字段 | 形状 |
|------|---------|------|
| `chain_embeddings.npz` | `chain_names`（字符串数组）、`embeddings`（float32） | `(m, d)` |
| `l0_embedding.npz` | `chain_names`（单元素数组）、`embeddings`（float32） | `(1, d)` |
| `chain_type_classification.csv` | `chain_name`、`type`（取值 `A` 或 `B`） | — |
| `source_part_*.npz` | `ids`（整数数组）、`embeddings`（float32） | `(n_batch, d)` |
| `source_part_*.parquet` | `id`（整数）、`source_text`（字符串） | — |

---

## 配置

将 `.env.example` 复制为 `.env` 并填写对应值：

```bash
cp .env.example .env
```

| 变量 | 说明 |
|------|------|
| `LLM_API_URL` | OpenAI 兼容的 LLM 接口地址（用于阈值标定） |
| `LLM_API_KEY` | API 鉴权密钥 |
| `LLM_MODEL_NAME` | LLM 判断所用的模型名称 |
| `EMBEDDING_MODEL_PATH` | 本地 embedding 模型路径 |
