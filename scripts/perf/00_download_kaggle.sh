#!/bin/bash
# Download IPL performance data from Kaggle
#
# Requires: kaggle CLI installed and configured with API key
# See: https://github.com/Kaggle/kaggle-api#api-credentials

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
KAGGLE_DIR="$BASE_DIR/data/perf/sources/kaggle"

mkdir -p "$KAGGLE_DIR"
cd "$KAGGLE_DIR"

echo "Downloading IPL performance data from Kaggle..."
echo "Target directory: $KAGGLE_DIR"
echo ""

# IPL dataset with ball-by-ball data
echo "=== Downloading IPL match data ==="
kaggle datasets download -d rahulaw2810/ipl-dataset
unzip -o ipl-dataset.zip -d ipl-dataset/
rm -f ipl-dataset.zip
echo "Done: ipl-dataset/"
echo ""

echo "=== All downloads complete ==="
ls -la "$KAGGLE_DIR"
