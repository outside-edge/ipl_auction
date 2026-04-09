# Data Directory

This directory contains all data for the IPL auction analysis pipeline.

## Directory Structure

```
data/
├── sources/          # Raw external data (read-only)
├── auction/          # Processed auction data
├── perf/             # Performance metrics
│   ├── ipl/          # IPL stats and WAR
│   └── t20i/         # T20I stats and WAR
└── analysis/         # Analysis outputs
    ├── joined/       # Merged datasets
    ├── diagnostics/  # Data quality reports
    └── predictions/  # Model outputs
```

## Key Files

### Auction Data (`auction/`)
- `auction_all_years.csv` - All auction records 2008-2026
- `player_registry.csv` - Canonical player IDs
- `name_aliases.csv` - Manual name corrections

### Performance Data (`perf/ipl/`)
- `player_season_stats.csv` - Batting/bowling by player-season
- `player_season_war.csv` - WAR values

### Analysis (`analysis/joined/`)
- `auction_with_performance.csv` - Auction + performance joined
- `auction_inflation_adjusted.csv` - Prices in 2024 rupees

## Pipeline

Run `make all` from project root to regenerate all data.

Individual targets:
- `make data` - Build auction and performance data
- `make integration` - Match names and adjust inflation
- `make analysis` - Run regressions and predictions
- `make diagnostics` - Generate unmatched player reports
