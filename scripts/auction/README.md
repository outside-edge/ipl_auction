# Auction Data Scripts

Scripts for assembling and managing IPL auction data from various sources.

## Pipeline Order

1. `01_scrape_auction_2026.py` - Scrape latest auction data from Wikipedia
2. `02_assemble_auction_data.py` - Combine all auction sources into unified dataset
3. `03_build_player_registry.py` - Create canonical player IDs and name aliases

## Outputs

All outputs go to `data/auction/`:

| File | Description |
|------|-------------|
| `auction_all_years.csv` | Unified auction data (2008-2026) |
| `player_registry.csv` | Player IDs and canonical names |
| `name_aliases.csv` | Manual name mappings (user-editable) |
| `retained_players.csv` | Pre-auction retained players |

## Data Sources

Input data from `data/sources/`:

- `kaggle/iplauctiondata/` - Historical auction data (2008-2021)
- `kaggle/IPL-2024-SOLD-PLAYER-DATA-ANALYSIS/` - 2024 auction
- `2022_auction/`, `2023_auction/` - Year-specific auction files
- `scraped/auction_2026.csv` - Wikipedia scrape for 2026
