# IPL Auction Analysis: Where Do the Crores Go?

## The Big Picture

IPL teams spent **₹687 Cr** on disappointing players from 2012-2025. Using ball-by-ball data to compute WAR (Wins Above Replacement), we identify which purchases wasted the most money.

## Top Disappointments (by Wasted Money)

<!-- TABLE:disappointments:start -->
**Total wasted: ₹687 Cr** across 378 disappointing players

| Year | Player | Team | Paid | Predicted | Actual | Wasted |
| --- | --- | --- | --- | --- | --- | --- |
| 2025 | Ravi Bishnoi | LSG | ₹11.0Cr | 7.8 | -0.7 | ₹8.6Cr |
| 2014 | Gautam Gambhir | KKR | ₹12.5Cr | 10.9 | 2.6 | ₹8.4Cr |
| 2025 | Shivam Dube | CSK | ₹12.0Cr | 7.2 | -1.1 | ₹8.4Cr |
| 2025 | Tilak Varma | MI | ₹8.0Cr | 10.5 | 2.3 | ₹8.0Cr |
| 2020 | Glenn Maxwell | PBKS | ₹10.8Cr | 8.4 | 1.0 | ₹7.5Cr |
| 2014 | Chris Gayle | RCB | ₹7.5Cr | 12.2 | 0.3 | ₹7.5Cr |
| 2015 | Yuvraj Singh | DC | ₹16.0Cr | 9.2 | 2.1 | ₹7.2Cr |
| 2018 | Chris Morris | DC | ₹11.0Cr | 7.8 | 0.7 | ₹7.2Cr |
| 2025 | Sanju Samson | RR | ₹18.0Cr | 10.2 | 3.2 | ₹7.2Cr |
| 2014 | Virat Kohli | RCB | ₹12.5Cr | 12.0 | 5.0 | ₹7.1Cr |
| 2014 | Shane Watson | RR | ₹12.5Cr | 13.5 | 6.6 | ₹7.0Cr |
| 2022 | Mohammed Siraj | RCB | ₹7.0Cr | 7.6 | -3.7 | ₹7.0Cr |
| 2018 | Manish Pandey | SRH | ₹11.0Cr | 6.6 | 0.0 | ₹6.7Cr |
| 2022 | Anrich Nortje | DC | ₹6.5Cr | 6.0 | -0.9 | ₹6.5Cr |
| 2014 | Shikhar Dhawan | SRH | ₹12.5Cr | 10.1 | 3.7 | ₹6.5Cr |
<!-- TABLE:disappointments:end -->

---

## Team Efficiency

<!-- TABLE:team_efficiency:start -->
| Team | N | Spent | WAR | WAR/Cr |
| --- | --- | --- | --- | --- |
| KKR | 74 | ₹281Cr | 295 | 1.05 |
| SRH | 78 | ₹311Cr | 306 | 0.99 |
| PBKS | 76 | ₹285Cr | 281 | 0.99 |
| RCB | 69 | ₹300Cr | 280 | 0.93 |
| CSK | 60 | ₹313Cr | 291 | 0.93 |
| RR | 72 | ₹310Cr | 269 | 0.87 |
| DC | 83 | ₹348Cr | 293 | 0.84 |
| MI | 68 | ₹330Cr | 258 | 0.78 |
| LSG | 25 | ₹148Cr | 114 | 0.77 |
<!-- TABLE:team_efficiency:end -->

---

## Backtest Results

How well can we predict player performance at auction time using only prior data?

<!-- TABLE:backtest_summary:start -->
| Year | N | R² | Rank ρ | RMSE |
| --- | --- | --- | --- | --- |
| 2012 | 10 | -0.22 | 0.56 | 4.9 |
| 2013 | 14 | -1.29 | -0.26 | 5.2 |
| 2014 | 125 | -0.05 | 0.39 | 4.9 |
| 2015 | 17 | -0.12 | 0.17 | 3.9 |
| 2016 | 43 | 0.01 | 0.17 | 3.9 |
| 2017 | 24 | -0.28 | -0.02 | 4.3 |
| 2018 | 116 | 0.14 | 0.32 | 5.2 |
| 2019 | 22 | 0.17 | 0.31 | 3.7 |
| 2020 | 28 | -0.18 | 0.31 | 4.6 |
| 2021 | 31 | -0.00 | 0.41 | 3.8 |
| 2022 | 150 | 0.12 | 0.37 | 5.3 |
| 2023 | 33 | 0.02 | 0.31 | 5.8 |
| 2025 | 43 | 0.29 | 0.55 | 4.8 |
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
