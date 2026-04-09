# Analysis Scripts

Scripts for joining auction and performance data, running regressions, and generating predictions.

## Pipeline Order

### Data Integration
1. `01_match_player_names.py` - Join auction to performance data

### Forecasting Pipeline (optional, run `make forecast`)
2. `02_build_player_master.py` - Link player identities across sources
3. `03_build_auction_features.py` - Create features for WAR prediction
4. `04_train_war_forecast.py` - Train XGBoost model for WAR forecasting

### Valuation Analysis
5. `05_hedonic_regression.py` - Hedonic wage regression
6. `06_identify_duds.py` - Find worst-value acquisitions
7. `07_predict_duds.py` - Predict overpays in upcoming auction

### Verification
8. `08_verify_data_consistency.py` - Data quality checks
9. `09_generate_diagnostics.py` - Unmatched player reports

### Utilities
- `utils.py` - Shared functions (inflation adjustment, etc.)

## Key Design Decisions

**Inflation adjustment is computed inline** in analysis scripts rather than saved as an intermediate file. This avoids storing redundant data since the computation is trivial.

## Outputs

### Joined Data (`data/analysis/joined/`)

| File | Description |
|------|-------------|
| `player_master.csv` | Unified player identities (for forecasting) |
| `auction_with_performance.csv` | Auction records matched to performance |
| `auction_features.csv` | Features for WAR prediction (forecasting pipeline) |

### Diagnostics (`data/analysis/diagnostics/`)

| File | Description |
|------|-------------|
| `unmatched_auction_by_year.csv` | Auction players without performance match |
| `unmatched_perf_by_year.csv` | Performance players without auction match |
| `match_rates_summary.txt` | Match rate statistics |
| `verification_report.md` | Data consistency report |

### Final Outputs (`tabs/`)

| File | Description |
|------|-------------|
| `regression_results.txt` | Hedonic regression coefficients |
| `worst_bets.csv` | Historical worst-value acquisitions |
| `predicted_duds_2026.csv` | Predicted overpays for 2026 |
