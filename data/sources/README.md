# Data Sources

Raw external data used as inputs to the pipeline. These files should be treated as read-only.

## Kaggle Datasets (`kaggle/`)

### iplauctiondata/
Historical IPL auction data (2008-2021).
- `IPLPlayerAuctionData.csv` - Player auction records

### IPL-2024-SOLD-PLAYER-DATA-ANALYSIS/
IPL 2024 auction data.
- `IPL 2024 SOLD PLAYER DATA ANALYSIS.csv` - 2024 sold players

### IPL-Data-viz/
Ball-by-ball IPL match data.
- `deliveries.csv` - Every ball bowled
- `matches.csv` - Match metadata

## Year-Specific Auctions

### 2022_auction/
- `IPL_2022_Sold_Players.csv`
- `IPL2022_Player_Auction_List.csv`
- `IPL_Auction_2022_FullList.csv`

### 2023_auction/
- `IPL_2023_Auction_Sold.csv`
- `IPL_2023_Auction_Pool.csv`
- `IPL_2023_Auction_Submitted.csv`

## Scraped Data (`scraped/`)

Data scraped from web sources.
- `auction_2026.csv` - Wikipedia scrape for 2026 mini auction

## Adding New Data

1. Place raw files in appropriate subdirectory
2. Update `scripts/auction/assemble_auction_data.py` if adding auction data
3. Run `make data` to regenerate processed files
