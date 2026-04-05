# IPL Auction Price Analysis (2008-2026)

## Predicted Duds for 2026

Which 2026 auction buys are most likely to disappoint? Based on the gap between price paid and model-predicted fair value:

| Rank | Player | Team | Price (Cr) | Prior WAR | Predicted (Cr) | Premium % |
|------|--------|------|------------|-----------|----------------|-----------|
| 1 | Liam Livingstone | SRH | ₹13.00 | 2.23 | ₹2.21 | +488% |
| 2 | Matheesha Pathirana | KKR | ₹18.00 | 7.42 | ₹3.40 | +429% |
| 3 | Ravi Bishnoi | RR | ₹7.20 | 0.45 | ₹1.51 | +378% |
| 4 | Venkatesh Iyer | RCB | ₹7.00 | 1.23 | ₹1.61 | +336% |
| 5 | Cameron Green | KKR | ₹25.20 | 13.86 | ₹5.81 | +334% |
| 6 | Mustafizur Rahman | KKR | ₹9.20 | 3.19 | ₹2.39 | +285% |
| 7 | Rahul Chahar | CSK | ₹5.20 | 0.17 | ₹1.47 | +254% |
| 8 | Josh Inglis | LSG | ₹8.60 | 7.42 | ₹3.40 | +153% |

**Methodology:** Uses most recent available WAR as prior performance. Premium % = how much teams overpaid relative to what the regression model predicts. High premiums historically correlate with underperformance.

**Caveat:** Predictions based on statistical patterns. Teams may have private information justifying higher prices. Also excludes 56 players new to IPL (no prior WAR data).

See `data/analysis/predicted_duds_2026.csv` for the full ranking.

---

## The Dumbest IPL Buys Ever

Which players cost teams the most while delivering the least? Here are the Top 10 worst bets in IPL auction history, ranked by absolute overpayment in crores:

| Rank | Player | Year | Team | Price (Cr) | Overpaid (Cr) | Premium % | WAR Shortfall |
|------|--------|------|------|------------|---------------|-----------|---------------|
| 1 | Gautam Gambhir | 2011 | KKR | ₹30.57 | ₹26.12 | +587% | -9.9 |
| 2 | Yuvraj Singh | 2015 | DC | ₹24.43 | ₹20.89 | +589% | +6.2 |
| 3 | Mitchell Starc | 2024 | KKR | ₹24.75 | ₹19.52 | +373% | +11.3 |
| 4 | Yuvraj Singh | 2014 | RCB | ₹22.46 | ₹17.52 | +355% | +8.3 |
| 5 | Pat Cummins | 2024 | SRH | ₹20.50 | ₹17.44 | +570% | -11.4 |
| 6 | Sam Curran | 2023 | PBKS | ₹19.33 | ₹16.72 | +642% | -2.6 |
| 7 | Pat Cummins | 2020 | KKR | ₹19.18 | ₹14.90 | +348% | +3.8 |
| 8 | Dinesh Karthik | 2014 | DC | ₹20.05 | ₹14.30 | +248% | +12.4 |
| 9 | Ravindra Jadeja | 2012 | CSK | ₹18.32 | ₹14.00 | +325% | +2.6 |
| 10 | Glenn Maxwell | 2021 | RCB | ₹16.45 | ₹13.43 | +445% | -20.3 |

**What this means:**
- **Overpaid (Cr)**: Absolute rupees wasted = Price Paid - Predicted Fair Price (in 2024 ₹)
- **Premium %**: Percentage overpayment relative to predicted price
- **WAR Shortfall**: Gap between prior-year WAR and next-season WAR (positive = underperformed expectations)

Gautam Gambhir tops the list: KKR paid ₹30.57 Cr in 2011 (inflation-adjusted), overpaying by ₹26.12 Cr relative to what his prior performance justified. Note that some "overpaid" players actually exceeded expectations (negative shortfall) — ranking by absolute overpayment captures the biggest financial gambles, regardless of outcome.

See `data/analysis/worst_bets.csv` for the full ranking of 257 player-seasons.

---

## Key Findings

Are IPL players paid according to their performance? This project applies hedonic wage regression models from labor economics to analyze the relationship between player auction prices and on-field performance.

| Finding | Estimate |
|---------|----------|
| Performance explains price variation | ~39-42% (R-squared) |
| 100 additional runs | ~45-60% price increase |
| 10 additional wickets | ~80-170% price increase |
| 1 WAR (Wins Above Replacement) | ~7% price increase |
| Overseas player premium | ~50-80% over Indian players |
| Prices predict future performance | Yes (p < 0.01) |

The overseas premium reflects scarcity value: each team is limited to 4 overseas players.

## Data Sources

| Source | Years | Description |
|--------|-------|-------------|
| Manual entry (NDTV, IPL official) | 2008 | Inaugural auction data |
| Wikipedia scraped | 2009-2015 | Historical auction prices |
| [Kaggle: iplauctiondata](https://github.com/paulramsey/iplauctiondata) | 2013-2022 | Main auction dataset |
| Official auction CSVs | 2021-2025 | Recent auction data |
| Wikipedia scraped | 2026 | Mini auction data |
| [Kaggle: ipl-dataset](https://www.kaggle.com/datasets) | 2008-2024 | Ball-by-ball match data |
| World Bank/RBI | 2008-2025 | India CPI for inflation adjustment |

## Project Structure

```
ipl_auction/
├── data/
│   ├── auction_all_years.csv          # Consolidated auction data (2,009 records)
│   ├── player_registry.csv            # Canonical player IDs + aliases (984 players)
│   ├── name_aliases.csv               # Manual name mappings for matching
│   ├── player_season_stats.csv        # Season performance stats (3,181 records)
│   ├── player_season_war.csv          # WAR (Wins Above Replacement) metrics
│   ├── auction_with_performance.csv   # Merged dataset
│   ├── india_cpi.csv                  # CPI reference (2008-2025)
│   ├── raw/
│   │   └── auction_2026.csv           # 2026 auction scraped data
│   └── analysis/
│       ├── auction_inflation_adjusted.csv
│       ├── regression_results.txt
│       ├── worst_bets.csv             # Ranked "dud" purchases
│       ├── predicted_duds_2026.csv    # 2026 auction dud predictions
│       └── fig_*.png                  # Visualizations
│
├── scripts/
│   ├── scrape_auction_2026.py         # Scrape 2026 auction from Wikipedia
│   ├── assemble_auction_data.py       # Consolidate auction sources
│   ├── build_player_registry.py       # Build player ID registry
│   ├── process_deliveries.py          # Ball-by-ball -> season stats
│   ├── compute_war.py                 # Ball-by-ball -> WAR metrics
│   ├── match_player_names.py          # Name matching across datasets
│   ├── adjust_inflation.py            # CPI adjustment to 2024 INR
│   ├── verify_data_consistency.py     # Data quality checks
│   ├── hedonic_regression.py          # Wage regression models
│   ├── identify_duds.py               # Identify worst auction bets
│   └── predict_duds.py                # Predict 2026 auction duds
│
├── notebooks/
│   ├── 01_descriptive_analysis.ipynb  # EDA & visualizations
│   └── 02_panel_analysis.ipynb        # Panel econometrics
│
├── Makefile                           # Pipeline automation
└── README.md
```

## Data Pipeline

```
Raw Sources               Processing                    Analysis
─────────────────────────────────────────────────────────────────────
Kaggle auction data  ─┐                              ┌─► auction_all_years.csv
Wikipedia scrape     ─┼─► assemble_auction_data.py ──┤           │
Official CSVs        ─┤   (deduplication +           └─► player_registry.csv
Manual 2008 entry    ─┤    player ID assignment)              │
2026 Wikipedia ───────┘  (scrape_auction_2026.py)             │
                                                              │
Ball-by-ball data    ─┬─► process_deliveries.py ─► player_season_stats.csv
                      │                                       │
                      └─► compute_war.py ────────► player_season_war.csv
                                                              │
                                                              ▼
                                               match_player_names.py
                                                              │
                                                              ▼
                                               auction_with_performance.csv
                                                              │
                                               verify_data_consistency.py ─► verification_report.md
                                                              │
CPI data             ─────────────────────────► adjust_inflation.py
                                                              │
                                                              ▼
                                               hedonic_regression.py
                                                              │
                                                              ▼
                                               regression_results.txt
                                               fig_*.png
                                                              │
                                               ┌──────────────┴──────────────┐
                                               ▼                              ▼
                                         identify_duds.py              predict_duds.py
                                               │                              │
                                               ▼                              ▼
                                         worst_bets.csv            predicted_duds_2026.csv
```

## Player Deduplication

The pipeline resolves player identity across years and sources:

| Issue | Example | Resolution |
|-------|---------|------------|
| Same-year spelling variants | "Nicolas Pooran" vs "Nicholas Pooran" (2022) | Fuzzy matching, keep priority source |
| Cross-year name changes | "Benjamin Stokes" (2017) vs "Ben Stokes" (2023) | Canonical `player_id` links all appearances |
| Nickname equivalents | "Chris Morris" vs "Christopher Morris" | NAME_EQUIVALENTS lookup |

Each player receives a unique `player_id` (P0001-P0984) enabling queries like:
"How many times was Ben Stokes auctioned?" → 3 times (2017, 2018, 2023)

## Methodology

This analysis applies the **hedonic wage model** (Rosen, 1974) to cricket labor markets, building on sports economics literature:

- **Scully (1974)**: Pioneered marginal revenue product estimation in baseball
- **Rastogi & Deodhar (2009)**: Applied hedonic pricing to early IPL auctions

### WAR (Wins Above Replacement)

We compute context-adjusted WAR metrics from ball-by-ball data:

- **Batting WAR**: (actual runs - replacement-level runs) / 8 runs per win
- **Bowling WAR**: (replacement-level runs - actual runs conceded) / 8 runs per win
- Replacement level: 15th percentile strike rate (batting), 80th percentile economy (bowling)

WAR provides a replacement-level-normalized metric that accounts for opportunities faced.

### Models Estimated

**Primary Models (no selection bias):**
1. **Lagged Performance Model**: log(price_t) ~ performance_{t-1} + controls
2. **WAR Lagged Model**: log(price_t) ~ WAR_{t-1} + controls

Lagged models are preferred because:
- Teams bid based on information available at auction time (prior performance)
- No selection on dependent variable (player doesn't need to play THIS season)
- All auctioned players can be included

**Secondary Models (for comparison):**
3. **Baseline OLS**: log(price) ~ runs + wickets + is_indian
4. **Full Model**: Adds batting average, strike rate, economy, catches
5. **WAR Current Model**: log(price) ~ total_war + is_indian
6. **Year Fixed Effects**: Controls for inflation and auction dynamics
7. **Market Efficiency Test**: future_WAR ~ log(price) + controls

All models use heteroscedasticity-robust standard errors (HC1).

### Interpreting Unexplained Variance

The models explain ~40% of price variation. The unexplained ~60% reflects multiple factors:

1. **Forecast error**: Teams buy expected FUTURE performance, not past. The gap between current-season R² (39%) and lagged R² (25%) confirms teams have private information about player trajectories.

2. **Auction mechanics**: Different slots/rounds, purse constraints, and team composition needs create price variation orthogonal to individual quality.

3. **Measurement error**: Raw runs/wickets don't capture context, consistency, or match impact. WAR addresses some but not all of this.

4. **Private information**: Teams have scouting insights, injury information, and fitness assessments we cannot observe.

The market efficiency test shows prices DO predict future performance (p < 0.01), suggesting teams have useful forecasting ability beyond what public stats reveal.

**Note**: This is NOT simply "star power" or "marketability." Player fixed effects absorb time-invariant characteristics, but even within-player variation shows substantial unexplained residuals due to the factors above.

## Usage

### Full Pipeline

```bash
# Step 1: Data Collection
python scripts/scrape_auction_2026.py

# Step 2: Data Assembly
python scripts/assemble_auction_data.py      # → auction_all_years.csv, player_registry.csv
python scripts/process_deliveries.py         # → player_season_stats.csv
python scripts/compute_war.py                # → player_season_war.csv

# Step 3: Data Integration
python scripts/match_player_names.py         # → auction_with_performance.csv
python scripts/adjust_inflation.py           # → auction_inflation_adjusted.csv

# Step 4: Verification
python scripts/verify_data_consistency.py    # → verification_report.md

# Step 5: Analysis
python scripts/hedonic_regression.py
python scripts/identify_duds.py
python scripts/predict_duds.py
```

Or use the Makefile:

```bash
make all           # Run full pipeline
make data          # Data collection and assembly only
make analysis      # Run regressions and generate predictions
make verify        # Data consistency checks
make clean         # Remove generated files
```

### Interactive Analysis

```bash
jupyter notebook notebooks/01_descriptive_analysis.ipynb
```

## Requirements

- Python 3.8+
- pandas
- numpy
- matplotlib
- seaborn
- statsmodels
- linearmodels (for panel models)

Install with:
```bash
pip install pandas numpy matplotlib seaborn statsmodels linearmodels beautifulsoup4
```

## Data Quality Notes

- **2,009 auction records** across 2008-2026 (after deduplication)
- **984 unique players** tracked via `player_id` across all years
- **33 players** have multiple name aliases resolved (e.g., "Ben Stokes"/"Benjamin Stokes")
- 3,181 player-season performance records
- 3,137 player-season WAR records
- Name matching rate: ~61% between auction and performance data
- Prices adjusted to 2024 INR using India CPI

## Limitations

### Selection Bias in Performance Matching

~39% of auctioned players have no same-season performance data because:
- Never fielded that season (injury, dropped, bench role)
- New to IPL (no prior performance)

This creates potential **selection bias** in same-season hedonic wage regressions. The current analysis:
- Uses lagged performance as primary specification to avoid conditioning on future outcomes
- Reports match rates by year for transparency
- Future work: Heckman correction or bounds analysis

## References

- Rosen, S. (1974). Hedonic prices and implicit markets. *Journal of Political Economy*, 82(1), 34-55.
- Scully, G. W. (1974). Pay and performance in Major League Baseball. *American Economic Review*, 64(6), 915-930.
- Rastogi, S., & Deodhar, R. (2009). Player pricing and valuation of cricketing attributes: Exploring the IPL Twenty20 vision. *Vikalpa*, 34(2), 15-24.

## License

Data compiled from public sources for research purposes.
