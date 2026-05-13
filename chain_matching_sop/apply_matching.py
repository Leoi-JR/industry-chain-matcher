#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main script for applying matching logic
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from matching.apply_logic import apply_matching_logic


def main():
    """Main entry point for matching logic application"""
    apply_matching_logic(show_progress=True)


if __name__ == "__main__":
    main()

