# Input Data Directory

This directory contains the threshold configuration files required by `apply_matching.py`.

## Required Files

You must create the following files before running `apply_matching.py`.  
They can be generated automatically by the calibration pipeline (see below) or set manually.

### 1. threshold_spec.csv

Per-chain thresholds for specific chains (chains NOT ending with "其他").

**Format:**
```csv
chain_name,threshold
示例产业链——上游——关键原材料——材料A——具体子类,0.65
示例产业链——上游——核心零部件——结构件——具体子类,0.70
```

**Requirements:**
- Must include all "specific" chains (non-"其他")
- Threshold values are typically between 0.5 and 0.9
- Determine these values through the calibration pipeline after running `run_similarity.py`

### 2. threshold_other.csv

Per-chain thresholds for "other" chains (chains ending with "其他").

**Format:**
```csv
chain_name,threshold
示例产业链——上游——关键原材料——材料A——其他,0.60
示例产业链——上游——核心零部件——结构件——其他,0.65
```

**Requirements:**
- Must include all "other" chains (ending with "其他")
- Threshold values are typically between 0.5 and 0.9
- Determine these values through the calibration pipeline after running `run_similarity.py`

### 3. threshold_l0.json

Global threshold for L0 (main industry chain) matching.

**Format:**
```json
{
  "T_L0": 0.65
}
```

**Requirements:**
- Single float value
- Typically between 0.5 and 0.8
- Applies to all Type A chains

## How to Determine Thresholds

After running `run_similarity.py`, you have raw similarity scores in `../similarity_output/`.

**Recommended: use the calibration pipeline**

```bash
python chain_matching_sop/sample_for_calibration.py      # sample by similarity bin
python chain_matching_sop/calibrate_thresholds_llm.py    # LLM judgment
python chain_matching_sop/compute_compliance_stats.py    # aggregate compliance rates
python chain_matching_sop/export_thresholds.py           # write threshold files here
```

**Or set manually** by loading a batch and inspecting the distribution:

```python
import numpy as np

# Load a similarity batch
sim_spec = np.load('../similarity_output/batch_0/sim_spec.npy')

# Inspect distribution across ranges
ranges = [(0.55, 0.60), (0.60, 0.65), (0.65, 0.70)]
for low, high in ranges:
    mask = (sim_spec > low) & (sim_spec <= high)
    print(f"[{low}, {high}): {mask.sum()} entries")
```
