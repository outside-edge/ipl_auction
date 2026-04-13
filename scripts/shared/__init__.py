#!/usr/bin/env python3
"""
Shared utilities for IPL auction analysis.

Modules:
- war: WAR (Wins Above Replacement) computation
- names: Player name normalization
- validation: Multi-year forward validation
- io: Dataset I/O (parquet/CSV handling)
- constants: Standard conventions for missing data and joins
"""

from .war import (
    RUNS_PER_WIN,
    compute_batting_war,
    compute_bowling_war,
    combine_war,
    validate_war,
)
from .war_gam import (
    compute_batting_war_gam,
    compute_bowling_war_gam,
    validate_gam_war,
    infer_batting_position,
)
from .names import normalize_name, get_initials_last, get_last_name, names_compatible
from .io import save_dataset, load_dataset, dataset_exists
from .constants import (
    SUFFIX_AUCTION,
    SUFFIX_IPL,
    SUFFIX_T20I,
    SUFFIX_WAR,
    SUFFIX_PERF,
    COL_HAS_IPL_HISTORY,
    COL_HAS_T20I_HISTORY,
    COL_IPL_SEASONS_PLAYED,
)
