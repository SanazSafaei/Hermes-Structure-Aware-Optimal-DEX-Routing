#!/bin/bash

# Shell script to find 10 common coins and run Ford algorithm on each
# Usage: ./run_ford_on_common_coins.sh [block_number] [min_snapshots]

set -e  # Exit on any error

# Default values
BLOCK_NUMBER=${1:-22361602}  # Use first argument or default to 22361602
ALGORITHM="ford-single-source"
MAX_COINS=10

echo "=================================================="
echo "Finding Common Coins and Running Ford Algorithm"
echo "=================================================="
echo "Block Number: $BLOCK_NUMBER"
echo "Max Coins: $MAX_COINS"
echo "Algorithm: $ALGORITHM"
echo ""

# Step 1: Find common coins
echo "Step 1: Reading coins from CSV file..."
echo "--------------------------------------------------"

# Default CSV file path - user can override with environment variable
CSV_FILE=${CSV_FILE:-"data/coins.csv"}

# Check if CSV file exists
if [[ ! -f "$CSV_FILE" ]]; then
    echo "ERROR: CSV file not found at: $CSV_FILE"
    echo "Please create a CSV file with coin data or set CSV_FILE environment variable"
    echo "Expected format: coin_id,symbol (with header row)"
    exit 1
fi

echo "Reading coins from: $CSV_FILE"

# Read coins from CSV file (skip header row, take first MAX_COINS entries)
# Expected CSV format: coin_id,symbol
COIN_IDS=$(python3 -c "
import csv
import sys

csv_file = '$CSV_FILE'
max_coins = $MAX_COINS

try:
    coins = []
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header row
        
        for i, row in enumerate(reader):
            if i >= max_coins:
                break
            if len(row) >= 2:
                coin_id = row[0].strip()
                if coin_id:  # Only add non-empty entries
                    coins.append(coin_id)
    
    print(coins)
except Exception as e:
    print('ERROR reading CSV file:', e, file=sys.stderr)
    sys.exit(1)
")

# Check if reading was successful
if [[ $? -ne 0 ]]; then
    echo "ERROR: Failed to read coins from CSV file"
    exit 1
fi

echo "Successfully read coins: $COINS_OUTPUT"
echo ""

# # Step 2: Parse the coins and run algorithm
# echo "Step 2: Running $ALGORITHM algorithm on each coin..."
# echo "--------------------------------------------------"

# # Extract coin IDs from the output format: [('id1', 'symbol1'), ('id2', 'symbol2'), ...]
# # We'll use python to parse this properly
# COIN_IDS=$(python3 -c "
# import ast
# import sys

# # Parse the coins output
# coins_str = '''$COINS_OUTPUT'''
# try:
#     coins = ast.literal_eval(coins_str)
#     coin_ids = [coin[0] for coin in coins]  # Extract just the IDs
#     print(' '.join(coin_ids))
# except Exception as e:
#     print('ERROR parsing coins:', e, file=sys.stderr)
#     sys.exit(1)
# ")

# Check if parsing was successful
if [[ $? -ne 0 ]]; then
    echo "ERROR: Failed to parse coin IDs"
    exit 1
fi

echo "Extracted coin IDs: $COIN_IDS"
echo ""

# Counter for progress tracking
COUNTER=1
TOTAL_COINS=$(echo $COIN_IDS | wc -w)

# Run $ALGORITHM algorithm on each coin
for COIN_ID in $COIN_IDS; do
    echo "[$COUNTER/$TOTAL_COINS] Running $ALGORITHM algorithm on coin: $COIN_ID"
    echo "Command: python3 main.py run-algorithms --algorithm $ALGORITHM --coin-name $COIN_ID --block-number $BLOCK_NUMBER"
    
    # Run the algorithm and capture output
    if python3 main.py run-algorithms \
        --algorithm $ALGORITHM \
        --coin-name "$COIN_ID" \
        --block-number $BLOCK_NUMBER; then
        echo "✓ Successfully completed $ALGORITHM algorithm for coin: $COIN_ID"
    else
        echo "✗ Failed to run $ALGORITHM algorithm for coin: $COIN_ID"
    fi
    
    echo ""
    ((COUNTER++))
done

echo "=================================================="
echo "All algorithms completed!"
echo "=================================================="

echo "Script completed successfully!"