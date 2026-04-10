#!/bin/bash
# Download IPL auction data from Kaggle
#
# Requires: kaggle CLI installed and configured with API key
# See: https://github.com/Kaggle/kaggle-api#api-credentials

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
KAGGLE_DIR="$BASE_DIR/data/auction/sources/kaggle"

mkdir -p "$KAGGLE_DIR"
cd "$KAGGLE_DIR"

echo "Downloading IPL auction data from Kaggle..."
echo "Target directory: $KAGGLE_DIR"
echo ""

# 2013-2022 auction data
echo "=== Downloading 2013-2022 auction data ==="
kaggle datasets download -d kalilurrahman/ipl-player-auction-data
unzip -o ipl-player-auction-data.zip -d iplauctiondata/
rm -f ipl-player-auction-data.zip
echo "Done: iplauctiondata/"
echo ""

# 2024 auction data
echo "=== Downloading 2024 auction data ==="
kaggle datasets download -d rohan0301/ipl-2024-sold-player-data-analysis
unzip -o ipl-2024-sold-player-data-analysis.zip -d ipl-2024-auction/
rm -f ipl-2024-sold-player-data-analysis.zip
echo "Done: ipl-2024-auction/"
echo ""

# 2025 auction data
echo "=== Downloading 2025 auction data ==="
kaggle datasets download -d stm321/ipl-data-viz
unzip -o ipl-data-viz.zip -d ipl-2025-auction/
rm -f ipl-data-viz.zip
echo "Done: ipl-2025-auction/"
echo ""

echo "=== All downloads complete ==="
ls -la "$KAGGLE_DIR"
