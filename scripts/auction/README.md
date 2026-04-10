# Auction Data Scripts

Scripts for assembling and managing IPL auction data from various sources.

## Pipeline Order

1. `01_scrape_auction_2026.py` - Scrape latest auction data from Wikipedia
2. `02_assemble_auction_data.py` - Combine all auction sources into unified dataset
3. `03_build_player_registry.py` - Create canonical player IDs and name aliases

## Outputs

All outputs go to `data/acquisitions/`:

| File | Description |
|------|-------------|
| `auction_all_years.csv` | Unified auction data (2008-2026) |
| `player_registry.csv` | Player IDs and canonical names |
| `name_aliases.csv` | Manual name mappings (user-editable) |
| `retained_players.csv` | Pre-auction retained players |

## Data Sources

Input data from `data/acquisitions/sources/`:

- `kaggle/iplauctiondata/` - Historical auction data (2013-2022)
- `kaggle/ipl-2024-auction/` - 2024 auction
- `kaggle/ipl-2025-auction/` - 2025 auction
- `kaggle_2022/`, `kaggle_2023/` - Year-specific auction files
- `scraped/auction_2026.csv` - Wikipedia scrape for 2026
- `wikipedia/` - Wikipedia Excel files (2009-2015)
- `manual/` - Manual data (2008, 2021)
