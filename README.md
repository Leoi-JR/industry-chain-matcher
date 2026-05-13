# Industry Chain Matcher

A matrix-based system for automatically matching enterprise information (business scope, patents, etc.) to nodes in a predefined industry chain graph.

[算法文档（中文）](README_zh.md) | [中文 README](README_zh.md) | [**Algorithm Visual Guide**](https://leoi-jr.github.io/industry-chain-matcher/index.html)

---

## Overview

Given an industry chain graph with `m` nodes and a corpus of `n` enterprise documents, this system efficiently maps each document to the most relevant chain node(s) using dense vector similarity and a set of configurable matching rules.

**Key features:**
- Batch matrix operations (no per-record loops) — scales to millions of documents
- Dual chain logic: industry-specific chains require an L0 (root) similarity gate; general-purpose chains do not
- Exclusion logic: "catch-all" nodes only match when no sibling-specific node matches
- LLM-assisted threshold calibration pipeline
- Optional GPU acceleration via CuPy

---

## Algorithm at a Glance

Each document embedding `d` is compared against three sets of chain embeddings:

```
Sim_L0          = L0_Embed          @ D.T   →  (1,       n)
Sim_Spec        = Spec_Embeds       @ D.T   →  (m_spec,  n)
Sim_Other_Parent= OtherParent_Embeds@ D.T   →  (m_other, n)
```

Final match masks are computed as:

| Chain type | Condition |
|------------|-----------|
| Specific (Type A) | `sim > threshold` **AND** `sim_L0 > T_L0` |
| Specific (Type B) | `sim > threshold` (no L0 gate) |
| "Other" catch-all | parent sim > threshold **AND** type gate **AND** no sibling matched |

See the inline code comments and README for algorithm details.

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate sample data

```bash
python sample_data/generate_sample_data.py
```

This creates a fictional 13-node industry chain with 200 mock source documents and matching threshold files — everything needed to run the full pipeline.

### 3. Run the matching pipeline

```bash
# Prepare configuration matrices (run once per chain definition)
python chain_matching_sop/prepare_chain_config.py

# Step 1: Compute raw similarity scores for all source batches
python chain_matching_sop/run_similarity.py

# Step 2: Apply thresholds and matching logic
python chain_matching_sop/apply_matching.py

# Step 3: Export final results as Parquet
python chain_matching_sop/export_results.py
```

### 4. Inspect results

```python
import pandas as pd
df = pd.read_parquet("chain_matching_sop/results/final_matching_results.parquet")
print(df.columns.tolist())
# ['chain_id', 'chain_name', 'info_id', 'similarity', 'source_text']
print(df.head())
```

---

## Directory Structure

```
industry-chain-matcher/
│
├── sample_data/                    # Sample data (safe to commit)
│   ├── generate_sample_data.py     # One-shot generator — run this first
│   ├── chain_embeddings.npz        # Chain node embeddings (generated)
│   ├── l0_embedding.npz            # L0 root embedding (generated)
│   ├── chain_type_classification.csv
│   ├── chain_definitions.json      # Text definitions for LLM calibration
│   ├── source_embeddings/          # Source document embeddings (generated)
│   └── source_texts/               # Source document texts (generated)
│
├── chain_matching_sop/             # Core matching algorithm
│   ├── config/
│   │   ├── data_config.py          # All file paths and format specs
│   │   └── sop_config.py           # Algorithm parameters
│   ├── data_preparation/           # prepare_chain_config.py modules
│   ├── similarity/                 # run_similarity.py modules
│   ├── matching/                   # apply_matching.py modules
│   ├── result_export/              # export_results.py modules
│   ├── utils/                      # I/O and matrix utilities
│   ├── prompt/                     # LLM prompt templates
│   ├── input_data/                 # Threshold configuration (see below)
│   │   ├── threshold_l0.json
│   │   ├── threshold_spec.csv
│   │   └── threshold_other.csv
│   ├── prepare_chain_config.py     # Step 0: build config matrices
│   ├── run_similarity.py           # Step 1: compute similarity scores
│   ├── sample_for_calibration.py   # Step 1.5a: sample for LLM review
│   ├── calibrate_thresholds_llm.py # Step 1.5b: LLM threshold calibration
│   ├── compute_compliance_stats.py # Step 1.5c: aggregate compliance rates
│   ├── export_thresholds.py        # Step 1.5d: generate threshold files
│   ├── apply_matching.py           # Step 2: apply matching logic
│   └── export_results.py           # Step 3: export results
│
├── embedding_code/                 # Upstream: chain definition & embedding pipeline
│   ├── llm_define_chains.py              # Step A1: LLM-generate text definitions
│   ├── parse_chain_definitions.py        # Step A2: Format and structure definitions
│   ├── embed_chain_definitions.py        # Step A3: Embed chain definitions
│   ├── preprocess_source_texts.py        # Step B1: Preprocess source text data
│   ├── embed_source_texts.py             # Step B2: Embed source text documents
│   ├── embed_utils.py                    # Shared embedding utilities
│   └── config.yaml                       # Configuration for embedding scripts
│
├── .env.example                    # Environment variable template
├── requirements.txt
├── README.md
└── README_zh.md                    # Chinese README
```

---

## Using Your Own Data

To run on a real industry chain, you need to:

> **Hardware requirement**: The embedding pipeline (`embedding_code/`) requires a CUDA-capable GPU.
> It uses `flash_attention_2` + `float16` and will fail on CPU-only machines.
> The core matching pipeline (`chain_matching_sop/`) has no GPU requirement.

### Step A — Prepare chain embeddings (upstream pipeline)

1. Place your industry chain Excel file at `sample_data/sample_industry_chain.xlsx`
2. Configure `embedding_code/config.yaml`
3. Set your embedding model path:
   ```bash
   export EMBEDDING_MODEL_PATH=/path/to/your/embedding-model
   ```
4. Run in order:
   ```bash
   python embedding_code/llm_define_chains.py        # generate text definitions
   python embedding_code/parse_chain_definitions.py  # format definitions
   python embedding_code/embed_chain_definitions.py  # embed definitions
   ```

5. Update `chain_matching_sop/config/data_config.py` to point `CHAIN_EMBEDDINGS_PATH`, `L0_EMBEDDINGS_PATH`, and `CHAIN_TYPE_CLASSIFICATION_PATH` at your generated files.

### Step B — Prepare source document embeddings

```bash
python embedding_code/preprocess_source_texts.py   # clean and split source texts

# Embed source texts (default column names)
python embedding_code/embed_source_texts.py --input_file source_texts_split/batch_0.parquet

# Embed a different text column
python embedding_code/embed_source_texts.py \
  --input_file patents/batch_0.parquet \
  --text_column custom_text_field \
  --id_column patent_id \
  --output_dir patent_embeddings
```

> **File naming convention**: `embed_source_texts.py` derives the output filename from the input filename (e.g. `source_part_0.parquet` → `source_part_0_embeddings.npz`). The downstream `data_config.py` locates these files via `SOURCE_EMBEDDINGS_PATTERN = "source_part_*.npz"`, so input files must be named with the `source_part_` prefix. `preprocess_source_texts.py` uses this prefix by default. If you supply custom input files, follow the same naming convention or update `SOURCE_EMBEDDINGS_PATTERN` in `data_config.py` accordingly.

Update `SOURCE_EMBEDDINGS_DIR` in `data_config.py` to point to your embedding output directory.

### Step C — Calibrate thresholds (optional but recommended)

After running Step 1, use the LLM pipeline to determine per-chain similarity thresholds:

```bash
python chain_matching_sop/sample_for_calibration.py          # sample by similarity bin
python chain_matching_sop/calibrate_thresholds_llm.py           # LLM judgment
python chain_matching_sop/compute_compliance_stats.py   # compute compliance rates
python chain_matching_sop/export_thresholds.py  # generate threshold files
```

Or manually edit `chain_matching_sop/input_data/threshold_spec.csv`, `threshold_other.csv`, and `threshold_l0.json`.

### Data format reference

| File | Required keys | Shape |
|------|--------------|-------|
| `chain_embeddings.npz` | `chain_names` (str array), `embeddings` (float32) | `(m, d)` |
| `l0_embedding.npz` | `chain_names` (1-element array), `embeddings` (float32) | `(1, d)` |
| `chain_type_classification.csv` | `chain_name`, `type` (`A` or `B`) | — |
| `source_part_*.npz` | `ids` (int array), `embeddings` (float32) | `(n_batch, d)` |
| `source_part_*.parquet` | `id` (int), `source_text` (str) | — |

---

## Configuration

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```

| Variable | Description |
|----------|-------------|
| `LLM_API_URL` | OpenAI-compatible LLM endpoint (used for threshold calibration) |
| `LLM_API_KEY` | API authentication key |
| `LLM_MODEL_NAME` | Model name to use for LLM judgments |
| `EMBEDDING_MODEL_PATH` | Local path to the embedding model |
