#!/usr/bin/env python3
"""
Constants and conventions for the IPL auction analysis pipeline.

Missing Data Conventions
------------------------
- NaN = no data available (player never played in that format)
- 0 = played but had zero value (e.g., zero WAR is meaningful)
- has_X_history flags indicate data availability

Performance lag features (ipl_war_lag1, t20i_war_12m, etc.):
    Use NaN for players with no history. This distinguishes "no data"
    from "zero performance" which is meaningful.

Count features (matches_played, ipl_seasons_played, etc.):
    Fill with 0 after joins. A player with no history has 0 matches.

Binary flags:
    has_ipl_history = 1 if player has IPL performance data, else 0
    has_t20i_history = 1 if player has T20I performance data, else 0
    These flags allow models to learn different relationships for
    rookies vs experienced players.

Join Suffix Conventions
-----------------------
When merging DataFrames with overlapping column names, use these suffixes:

    _auction  - Columns from auction data
    _ipl      - Columns from IPL performance data
    _t20i     - Columns from T20I performance data
    _war      - Columns from WAR calculations
    _perf     - General performance data (when source is clear from context)

Never use default pandas suffixes (_x, _y) as they provide no semantic meaning.
"""

# WAR computation constants (also in war.py but exposed here for reference)
RUNS_PER_WIN = 8

# Replacement level percentiles
BATTING_REPLACEMENT_PCT = 0.15
BOWLING_REPLACEMENT_PCT = 0.80

# Minimum balls for WAR qualification
MIN_BALLS_BATTING = 30
MIN_BALLS_BOWLING = 60

# Join suffixes
SUFFIX_AUCTION = "_auction"
SUFFIX_IPL = "_ipl"
SUFFIX_T20I = "_t20i"
SUFFIX_WAR = "_war"
SUFFIX_PERF = "_perf"

# History flag column names
COL_HAS_IPL_HISTORY = "has_ipl_history"
COL_HAS_T20I_HISTORY = "has_t20i_history"
COL_IPL_SEASONS_PLAYED = "ipl_seasons_played"
