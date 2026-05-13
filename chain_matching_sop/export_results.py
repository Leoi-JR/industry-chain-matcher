#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export final matching results to Parquet.

Delegates all logic to result_export.result_exporter.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from result_export.result_exporter import export_final_results

if __name__ == "__main__":
    export_final_results()
