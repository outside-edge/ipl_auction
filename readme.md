# IPL Auction Analysis: Where Do the Crores Go?

## The Big Picture

IPL teams spent **₹687 Cr** on disappointing players from 2012-2025. Using ball-by-ball data to compute WAR (Wins Above Replacement), we identify which purchases wasted the most money.

## Top Disappointments (by Wasted Money)

<!-- TABLE:disappointments:start -->
**Total wasted: ₹538 Cr** across 350 disappointing players

| Year | Player | Team | Paid | Predicted | Actual | Wasted |
| --- | --- | --- | --- | --- | --- | --- |
| 2014 | MS Dhoni | CSK | ₹12.5Cr | 15.8 | 5.4 | ₹8.0Cr |
| 2014 | Chris Gayle | RCB | ₹7.5Cr | 10.7 | -0.2 | ₹7.5Cr |
| 2018 | Chris Morris | DC | ₹11.0Cr | 10.7 | 1.2 | ₹7.4Cr |
| 2022 | Mohammed Siraj | RCB | ₹7.0Cr | 5.4 | -3.3 | ₹6.7Cr |
| 2019 | Jaydev Unadkat | RR | ₹8.4Cr | 8.0 | -0.5 | ₹6.6Cr |
| 2018 | Manish Pandey | SRH | ₹11.0Cr | 5.8 | -2.6 | ₹6.5Cr |
| 2022 | Kieron Pollard | MI | ₹6.0Cr | 5.8 | -2.1 | ₹6.0Cr |
| 2020 | Glenn Maxwell | PBKS | ₹10.8Cr | 6.1 | -1.6 | ₹5.9Cr |
| 2022 | Ruturaj Gaikwad | CSK | ₹6.0Cr | 10.8 | 3.2 | ₹5.9Cr |
| 2014 | Virat Kohli | RCB | ₹12.5Cr | 11.5 | 3.9 | ₹5.9Cr |
| 2018 | Axar Patel | PBKS | ₹12.5Cr | 8.5 | 1.1 | ₹5.7Cr |
| 2018 | Marcus Stoinis | PBKS | ₹6.2Cr | 6.3 | -0.9 | ₹5.6Cr |
| 2014 | Jacques Kallis | KKR | ₹5.5Cr | 8.9 | 0.3 | ₹5.5Cr |
| 2014 | Kieron Pollard | MI | ₹7.5Cr | 8.0 | 1.0 | ₹5.4Cr |
| 2014 | Murali Vijay | DC | ₹5.0Cr | 13.8 | 0.1 | ₹5.0Cr |
<!-- TABLE:disappointments:end -->

---

## Team Efficiency

<!-- TABLE:team_efficiency:start -->
| Team | N | Spent | WAR | WAR/Cr |
| --- | --- | --- | --- | --- |
| KKR | 67 | ₹235Cr | 350 | 1.49 |
| SRH | 73 | ₹236Cr | 302 | 1.28 |
| RR | 66 | ₹231Cr | 293 | 1.27 |
| RCB | 66 | ₹263Cr | 332 | 1.26 |
| CSK | 55 | ₹248Cr | 293 | 1.18 |
| PBKS | 73 | ₹275Cr | 324 | 1.18 |
| LSG | 20 | ₹97Cr | 105 | 1.09 |
| DC | 78 | ₹303Cr | 309 | 1.02 |
| MI | 63 | ₹255Cr | 238 | 0.93 |
<!-- TABLE:team_efficiency:end -->

---

## Backtest Results

How well can we predict player performance at auction time using only prior data?

<!-- TABLE:backtest_summary:start -->
| Year | N | R² | Rank ρ | RMSE |
| --- | --- | --- | --- | --- |
| 2012 | 9 | 0.29 | 0.67 | 5.8 |
| 2013 | 14 | -0.05 | 0.32 | 7.6 |
| 2014 | 124 | -0.24 | 0.31 | 5.8 |
| 2015 | 17 | -0.13 | 0.28 | 4.6 |
| 2016 | 43 | 0.27 | 0.57 | 5.6 |
| 2017 | 24 | -0.38 | -0.27 | 7.7 |
| 2018 | 116 | 0.05 | 0.24 | 6.0 |
| 2019 | 21 | -0.03 | 0.11 | 5.0 |
| 2020 | 27 | -0.29 | -0.07 | 5.4 |
| 2021 | 31 | 0.05 | 0.33 | 4.7 |
| 2022 | 150 | 0.22 | 0.48 | 5.4 |
| 2023 | 33 | -0.18 | 0.02 | 6.9 |
<!-- TABLE:backtest_summary:end -->

---

## Methodology

### WAR Calculation

From 3,137 player-season records of ball-by-ball data:

**Batting WAR** = (runs - expected_runs - dismissal_penalty) / RUNS_PER_WIN
- Expected runs = balls_faced × replacement_strike_rate
- Dismissal penalty = dismissals × RUNS_PER_DISMISSAL (6.0)

**Bowling WAR** = (replacement_runs - runs_conceded + wicket_bonus) / RUNS_PER_WIN
- Replacement runs = overs × replacement_economy
- Wicket bonus = wickets × RUNS_PER_WICKET (6.0)

**Replacement levels**: 15th percentile SR (batting), 80th percentile economy (bowling)

### Empirically Estimated Constants

| Constant | Value | Derivation |
|----------|-------|------------|
| RUNS_PER_DISMISSAL | 6.0 | OLS regression: innings_runs ~ wickets_lost + overs_batted (n=2,677 T20I innings) |
| RUNS_PER_WICKET | 6.0 | Same regression; symmetric by assumption (see note below) |
| RUNS_PER_WIN | 10 | Logistic regression on match outcomes + close-game heuristic |

**Note on symmetry**: RUNS_PER_DISMISSAL and RUNS_PER_WICKET are conceptually distinct (batting opportunity cost vs. bowling value), but the innings-level regression provides a single estimate for the marginal effect of a wicket on team total (~6 runs). We apply this symmetrically: a dismissed batter costs their team ~6 potential runs; a bowler taking a wicket saves their team ~6 runs. This is a modeling choice that simplifies the framework while remaining empirically grounded.

### Wasted Money Calculation

For each player:
1. Predict WAR using only info available at auction (lagged IPL WAR, T20I history)
2. Compare to actual WAR
3. **Wasted = (predicted - actual) × price_per_WAR**, capped at price paid

Implied market price: **₹1.01 Cr per WAR**

### Prediction Model

XGBoost with expanding window training:
- Features: lag1/2/3 IPL WAR, T20I WAR, career totals, trends
- Training: all prior auction+outcome data
- Mean rank correlation: **0.28** (13 years evaluated)

---

## Repository Structure

```
ipl_auction/
├── data/
│   ├── acquisitions/         # Auction price data
│   └── perf/                 # IPL/T20I WAR by season
├── scripts/
│   ├── auction/              # Scraping, assembly
│   ├── perf/                 # Performance data processing
│   ├── prediction/           # WAR prediction, backtest
│   ├── retrospective/        # Hedonic regression
│   └── shared/               # Utilities
├── tabs/                     # Output tables (CSV)
│   ├── disappointments.csv   # Sorted by wasted money
│   ├── team_efficiency.csv
│   └── retroactive_*.csv
└── .github/workflows/        # Auto-update README
```

## Quick Start

```bash
# Full pipeline
make all

# Just analysis
python3 scripts/prediction/06_comprehensive_backtest.py
python3 scripts/prediction/07_economic_analysis.py

# Update README tables
python3 scripts/update_readme_tables.py
```

## Data Sources

| Source | Coverage |
|--------|----------|
| Kaggle, Wikipedia, IPL official | Auction prices 2008-2026 |
| Kaggle ball-by-ball | Match data 2008-2025 |
| World Bank/RBI | CPI for inflation adjustment |

## Limitations

1. **Missing years**: 2024 auction data not available
2. **Selection bias**: ~19% of auctioned players never fielded
3. **Private information**: Teams have scouting data we don't

## References

- Rosen, S. (1974). Hedonic prices and implicit markets. *JPE*.
- Scully, G. W. (1974). Pay and performance in MLB. *AER*.
