# Performance Data Scripts

Scripts for processing ball-by-ball cricket data into player performance metrics and WAR.

## Pipeline Order

### IPL Performance
1. `01_process_ipl_deliveries.py` - Process ball-by-ball data into season stats
2. `02_compute_ipl_war.py` - Calculate WAR (Wins Above Replacement)

### T20I Performance (for forecasting)
3. `03_download_t20i.py` - Download T20I data from Cricsheet
4. `04_process_t20i_deliveries.py` - Process T20I deliveries
5. `05_compute_t20i_war.py` - Calculate T20I WAR

## Outputs

### IPL (`data/perf/ipl/`)

| File | Description |
|------|-------------|
| `player_season_stats.csv` | Batting/bowling stats by player-season |
| `player_season_war.csv` | WAR values by player-season |

### T20I (`data/perf/t20i/`)

| File | Description |
|------|-------------|
| `deliveries.csv` | Processed ball-by-ball data |
| `matches.csv` | Match metadata |
| `registry.csv` | Player ID mappings |
| `player_year_war.csv` | WAR by player-year |

## Data Sources

- `data/perf/sources/kaggle/ipl-dataset/` - IPL ball-by-ball data
- Cricsheet T20I JSON files (downloaded automatically)
