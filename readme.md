# IPL Auction Analysis: Where Do the Crores Go?

## The Big Picture

IPL teams spent **₹1,854 Cr** on 297 players from 2012-2025. Using ball-by-ball data to compute WAR (Wins Above Replacement), we ask: did they get what they paid for?

### Key Findings

| Stat | Value |
|------|-------|
| Players who massively underperformed (5+ WAR miss) | 59 |
| Money spent on those underperformers | **₹442 Cr** |
| Players who massively outperformed (10+ WAR bonus) | 28 |
| Correlation: price vs actual performance | 0.32 |
| Correlation: price vs expected performance | 0.50 |

**The takeaway**: Teams pay for what they expect, not what they get. Expensive players are riskier bets—the 10+ Cr tier has 28% underperformance rate vs 12% for <2 Cr players.

---

## The Waste: ₹442 Crore on Disappointments

Players who delivered 5+ WAR below expectations:

| Player | Year | Price | Expected | Actual | Gap |
|--------|------|-------|----------|--------|-----|
| Chris Gayle | 2014 | 7.5 Cr | 14.9 WAR | 0.4 WAR | -14.5 |
| Daniel Christian | 2021 | 4.0 Cr | 9.2 WAR | -2.5 WAR | -11.6 |
| Brendon McCullum | 2012 | 4.5 Cr | 9.9 WAR | -0.6 WAR | -10.6 |
| Gautam Gambhir | 2014 | 12.5 Cr | 13.6 WAR | 3.3 WAR | -10.3 |
| Chris Morris | 2021 | 16.0 Cr | 8.2 WAR | -1.7 WAR | -9.9 |
| Shivam Dube | 2025 | 12.0 Cr | 8.4 WAR | -1.3 WAR | -9.8 |

---

## The Steals: Best Value Finds

Players who exceeded expectations at bargain prices:

| Player | Year | Price | Expected | Actual | Surplus per Cr |
|--------|------|-------|----------|--------|----------------|
| Jitesh Sharma | 2022 | 0.2 Cr | -0.5 WAR | 10.7 WAR | +56 WAR/Cr |
| Ajinkya Rahane | 2023 | 0.5 Cr | 0.0 WAR | 14.4 WAR | +29 WAR/Cr |
| Darren Bravo | 2012 | 0.5 Cr | 5.1 WAR | 16.7 WAR | +23 WAR/Cr |
| Sonu Yadav | 2023 | 0.2 Cr | 15.4 WAR | 29.1 WAR | +20 WAR/Cr |

---

## Breakout Stars: Biggest Positive Surprises

| Player | Year | Price | Expected | Actual | Bonus |
|--------|------|-------|----------|--------|-------|
| Jos Buttler | 2022 | 10.0 Cr | 7.1 WAR | 33.5 WAR | +26.4 |
| Liam Livingstone | 2022 | 11.5 Cr | 0.6 WAR | 24.0 WAR | +23.4 |
| Sunil Narine | 2018 | 12.5 Cr | 9.6 WAR | 31.6 WAR | +21.9 |
| Cameron Green | 2023 | 17.5 Cr | 0.4 WAR | 19.7 WAR | +19.3 |
| Heinrich Klaasen | 2023 | 5.2 Cr | 1.6 WAR | 20.2 WAR | +18.6 |
| Rishabh Pant | 2018 | 15.0 Cr | 11.3 WAR | 28.7 WAR | +17.4 |

---

## The Gamble: Expensive = Risky

| Price Tier | N | Avg Miss | Variance | Underperform Rate |
|------------|---|----------|----------|-------------------|
| <2 Cr | 102 | -0.2 WAR | 5.0 | 12% |
| 2-5 Cr | 53 | -1.1 WAR | 5.7 | 25% |
| 5-10 Cr | 70 | +1.9 WAR | 7.4 | 20% |
| **10+ Cr** | 72 | +1.6 WAR | **7.9** | **28%** |

Expensive players have higher variance. The 10+ Cr tier has almost 60% more variance than the <2 Cr tier.

---

## Team Efficiency: Who Finds Value?

WAR per crore spent (higher = more efficient, teams with 20+ players):

| Team | N | Total Spent | Actual WAR | WAR/Cr |
|------|---|-------------|------------|--------|
| KKR | 34 | 170 Cr | 221 | 1.30 |
| PBKS | 26 | 141 Cr | 168 | 1.19 |
| DC | 30 | 173 Cr | 180 | 1.04 |
| RCB | 27 | 189 Cr | 185 | 0.98 |
| RR | 35 | 225 Cr | 217 | 0.96 |
| SRH | 34 | 204 Cr | 196 | 0.96 |
| MI | 37 | 249 Cr | 230 | 0.92 |
| CSK | 31 | 238 Cr | 217 | 0.91 |

KKR outperforms: paid 25 Cr less than "fair" over this period. CSK and MI are the big spenders with below-average efficiency.

See `tabs/team_efficiency.csv` for full rankings.

---

## Mega vs Mini Auctions

| Auction Type | N | Mean Error | Std Dev | Price-Perf Corr |
|--------------|---|------------|---------|-----------------|
| Mega | 231 | +0.9 WAR | 6.5 | ρ = 0.52 |
| Mini | 66 | -0.5 WAR | 6.9 | ρ = 0.15 |

Mega auctions have tighter price-performance correlation (ρ = 0.52 vs 0.15). The reset gives teams better information.

---

## 2026 Dud Predictions

Most likely to disappoint:

| Player | Team | Price | Prior WAR | Fair Price | Premium |
|--------|------|-------|-----------|------------|---------|
| Cameron Green | KKR | 25.2 Cr | 13.9 | 5.1 Cr | +391% |
| Matheesha Pathirana | KKR | 18.0 Cr | 7.4 | 3.4 Cr | +433% |
| Liam Livingstone | SRH | 13.0 Cr | 2.2 | 2.4 Cr | +439% |
| Venkatesh Iyer | RCB | 7.0 Cr | 1.2 | 1.7 Cr | +312% |

---

## Methodology

### WAR Calculation

From 3,137 player-season records of ball-by-ball data:

- **Batting WAR** = (actual runs − replacement runs) / 8
- **Bowling WAR** = (replacement runs − actual runs conceded) / 8
- Replacement level: 15th percentile (batting), 80th percentile (bowling)

### Prediction Model

For each auction year (2012-2025):

1. **Features**: Lag 1/2/3 IPL WAR, T20I WAR, career totals, trends
2. **Training**: All prior auction+outcome data (expanding window)
3. **Model**: XGBoost with regularization
4. **Evaluation**: Rank correlation, R², price correlations

### Hedonic Regression

log(price) ~ lagged_WAR + is_indian + is_mega_auction + controls

| Variable | Effect on Price |
|----------|-----------------|
| +1 WAR (prior season) | +8.6% |
| Overseas player | +50% premium |
| Lagged performance R² | 36% |

---

## Repository Structure

```
ipl_auction/
├── data/
│   ├── acquisitions/         # Auction price data
│   │   ├── auction_all_years.parquet
│   │   ├── player_registry.csv
│   │   └── sources/          # Raw auction CSVs
│   ├── perf/
│   │   ├── ipl/              # IPL WAR by season
│   │   └── t20i/             # T20I WAR by year
│   ├── analysis/             # Merged datasets
│   └── model/                # Feature matrices
│
├── scripts/
│   ├── auction/              # Scraping, assembly
│   │   ├── 01_scrape_auction_2026.py
│   │   ├── 02_assemble_auction_data.py
│   │   └── 03_build_player_registry.py
│   │
│   ├── perf/                 # Performance data
│   │   ├── 01_process_ipl_deliveries.py
│   │   ├── 02_compute_ipl_war.py
│   │   └── 05_compute_t20i_war.py
│   │
│   ├── verify/               # Data quality
│   │   ├── 01_verify_data_consistency.py
│   │   ├── 02_match_player_names.py
│   │   └── 03_build_player_master.py
│   │
│   ├── retrospective/        # Oracle analysis (uses actual WAR)
│   │   ├── 01_hedonic_regression.py
│   │   └── 02_identify_duds.py
│   │
│   ├── prediction/           # Prospective analysis (lagged only)
│   │   ├── 01_build_auction_features.py
│   │   ├── 02_train_war_forecast.py
│   │   ├── 04_predict_duds.py
│   │   ├── 06_comprehensive_backtest.py  # Main backtest
│   │   └── 07_economic_analysis.py       # Newspaper stats
│   │
│   └── shared/               # Utilities
│       ├── war.py
│       ├── names.py
│       ├── inflation.py
│       └── io.py
│
├── tabs/                     # Output tables
│   ├── retroactive_predictions.csv
│   ├── retroactive_summary.csv
│   ├── economic_summary.txt
│   ├── team_efficiency.csv
│   ├── lucky_unlucky.csv
│   ├── worst_bets.csv
│   └── regression_results.txt
│
├── figs/                     # Visualizations
└── Makefile
```

## Quick Start

```bash
# Full pipeline
make all

# Just the economic analysis
python3 scripts/prediction/06_comprehensive_backtest.py
python3 scripts/prediction/07_economic_analysis.py

# View key results
cat tabs/economic_summary.txt
cat tabs/team_efficiency.csv
```

## Data Sources

| Source | Coverage |
|--------|----------|
| Wikipedia, NDTV, IPL official | Auction prices 2008-2026 |
| Kaggle ball-by-ball | Match data 2008-2025 |
| World Bank/RBI | CPI for inflation adjustment |

## Limitations

1. **Selection bias**: ~19% of auctioned players never fielded
2. **Supply dynamics**: Price depends on who else is available
3. **Private information**: Teams have scouting data we don't
4. **Salary caps**: Retained superstars hit ceiling prices

## References

- Rosen, S. (1974). Hedonic prices and implicit markets. *JPE*.
- Scully, G. W. (1974). Pay and performance in MLB. *AER*.
- Rastogi & Deodhar (2009). Player pricing in IPL. *Vikalpa*.
