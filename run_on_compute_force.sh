#!/bin/bash

ALG_NAME="$1"
TOP_COINS_COUNT="$2"
START_BLOCK="$3"
END_BLOCK="$4"
WORKERS="${5:-24}"

if [[ -z "$ALG_NAME" || -z "$TOP_COINS_COUNT" || -z "$START_BLOCK" || -z "$END_BLOCK" ]]; then
    echo "Usage: $0 <alg_name> <top_coins_count> <start_block> <end_block>"
    exit 1
fi

python3 -u main.py \
    --config config.compute.json \
    --log-level INFO \
    run \
    --alg "$ALG_NAME" \
    --start-block-number "$START_BLOCK" \
    --end-block-number "$END_BLOCK" \
    --top-coins-count "$TOP_COINS_COUNT" \
    --query-count 100 \
    --workers "$WORKERS" \
    --force
