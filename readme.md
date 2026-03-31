# IPL Auction Price Analysis (2008-2026)

## Predicted Duds for 2026

Which 2026 auction buys are most likely to disappoint? Based on the gap between price paid and model-predicted fair value:

| Rank | Player | Team | Price (Cr) | Prior WAR | Predicted (Cr) | Premium % |
|------|--------|------|------------|-----------|----------------|-----------|
| 1 | Liam Livingstone | SRH | ₹13.00 | 2.23 | ₹2.24 | +481% |
| 2 | Matheesha Pathirana | KKR | ₹18.00 | 7.42 | ₹3.44 | +424% |
| 3 | Ravi Bishnoi | RR | ₹7.20 | 0.45 | ₹1.50 | +380% |
| 4 | Venkatesh Iyer | RCB | ₹7.00 | 1.23 | ₹1.60 | +338% |
| 5 | Cameron Green | KKR | ₹25.20 | 13.86 | ₹5.86 | +330% |
| 6 | Mustafizur Rahman | KKR | ₹9.20 | 3.19 | ₹2.42 | +280% |
| 7 | Rahul Chahar | CSK | ₹5.20 | 0.17 | ₹1.47 | +255% |
| 8 | Josh Inglis | LSG | ₹8.60 | 7.42 | ₹3.44 | +150% |

**Methodology:** Uses most recent available WAR as prior performance. Premium % = how much teams overpaid relative to what the regression model predicts. High premiums historically correlate with underperformance.

**Caveat:** Predictions based on statistical patterns. Teams may have private information justifying higher prices. Also excludes 56 players new to IPL (no prior WAR data).

See `data/analysis/predicted_duds_2026.csv` for the full ranking.

---

## The Dumbest IPL Buys Ever

Which players cost teams the most while delivering the least? Here are the Top 10 worst bets in IPL auction history, ranked by overpayment combined with underperformance:

| Rank | Player | Year | Team | Price (Cr) | Premium Paid | WAR Shortfall |
|------|--------|------|------|------------|--------------|---------------|
| 1 | Mitchell Starc | 2024 | KKR | ₹24.75 | +462% | -11.3 |
| 2 | Jaydev Unadkat | 2018 | RR | ₹15.62 | +232% | -15.6 |
| 3 | Yuvraj Singh | 2015 | DC | ₹24.43 | +753% | -6.2 |
| 4 | Yuvraj Singh | 2014 | RCB | ₹22.46 | +373% | -8.3 |
| 5 | Shane Watson | 2016 | RCB | ₹13.82 | +267% | -9.5 |
| 6 | Ashish Nehra | 2016 | SRH | ₹8.00 | +160% | -13.7 |
| 7 | Robin Uthappa | 2011 | PWI | ₹18.47 | +139% | -16.0 |
| 8 | Kedar Jadhav | 2018 | CSK | ₹10.59 | +154% | -11.7 |
| 9 | Saurabh Tiwary | 2011 | MI | ₹13.85 | +132% | -13.9 |
| 10 | Mohit Sharma | 2016 | PBK | ₹9.46 | +357% | -5.2 |

**What this means:**
- **Premium Paid**: How much the team overpaid relative to what performance-based models predicted
- **WAR Shortfall**: The gap between prior-year WAR and next-season WAR (negative = player performed worse than their track record)
- **Dud Score**: Combines both factors — you need to both overpay AND underdeliver to top this list

Mitchell Starc tops the list despite being a world-class bowler: KKR paid ₹24.75 Cr (a 462% premium over expected price), and his 2024 WAR dropped by 11.3 from his prior season. The "dumbest" buys aren't necessarily bad players — they're cases where team expectations (reflected in price) wildly exceeded actual performance.

See `data/analysis/worst_bets.csv` for the full ranking of 250 player-seasons.

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
│   ├── auction_all_years.csv          # Consolidated auction data (2,025 records)
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
│   ├── assemble_auction_data.py       # Consolidate auction sources
│   ├── scrape_auction_2026.py         # Scrape 2026 auction from Wikipedia
│   ├── process_deliveries.py          # Ball-by-ball -> season stats
│   ├── compute_war.py                 # Ball-by-ball -> WAR metrics
│   ├── match_player_names.py          # Name matching across datasets
│   ├── adjust_inflation.py            # CPI adjustment to 2024 INR
│   ├── hedonic_regression.py          # Wage regression models
│   ├── identify_duds.py               # Identify worst auction bets
│   └── predict_duds.py                # Predict 2026 auction duds
│
├── notebooks/
│   ├── 01_descriptive_analysis.ipynb  # EDA & visualizations
│   └── 02_panel_analysis.ipynb        # Panel econometrics
│
└── README.md
```

## Data Pipeline

```
Raw Sources               Processing                    Analysis
─────────────────────────────────────────────────────────────────────
Kaggle auction data  ─┐
Wikipedia scrape     ─┼─► assemble_auction_data.py ─► auction_all_years.csv
Official CSVs        ─┤                                       │
Manual 2008 entry    ─┤                                       │
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

1. **Baseline OLS**: log(price) ~ runs + wickets + is_indian
2. **Full Model**: Adds batting average, strike rate, economy, catches
3. **WAR Model**: log(price) ~ total_war + is_indian
4. **Lagged Model**: Uses prior season performance as predictors
5. **Year Fixed Effects**: Controls for inflation and auction dynamics
6. **Player Fixed Effects**: Controls for time-invariant player characteristics
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

```bash
# Run pipeline in order
python scripts/scrape_auction_2026.py        # Scrape 2026 auction data
python scripts/assemble_auction_data.py      # Consolidate all auction sources
python scripts/process_deliveries.py
python scripts/compute_war.py
python scripts/match_player_names.py
python scripts/adjust_inflation.py
python scripts/hedonic_regression.py
python scripts/identify_duds.py              # Historical worst bets
python scripts/predict_duds.py               # 2026 predictions

# Or run notebooks for interactive analysis
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

- 2,025 auction records across 2008-2026
- 3,181 player-season performance records
- 3,137 player-season WAR records
- Name matching rate: ~62% between auction and performance data
- Prices adjusted to 2024 INR using India CPI
- Deduplication prioritizes official sources over scraped data

## References

- Rosen, S. (1974). Hedonic prices and implicit markets. *Journal of Political Economy*, 82(1), 34-55.
- Scully, G. W. (1974). Pay and performance in Major League Baseball. *American Economic Review*, 64(6), 915-930.
- Rastogi, S., & Deodhar, R. (2009). Player pricing and valuation of cricketing attributes: Exploring the IPL Twenty20 vision. *Vikalpa*, 34(2), 15-24.

## License

Data compiled from public sources for research purposes.
