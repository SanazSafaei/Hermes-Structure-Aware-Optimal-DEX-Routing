import os
import glob
from collections import Counter, defaultdict
import pandas as pd
import logging

def find_common_coins(data_dir="data/snapshots", min_snapshots=None, max_coins=10):
    """Find coins that appear in multiple snapshots, sorted by highest totalValueLockedUSD"""
    
    # Get all snapshot files
    snapshot_files = glob.glob(os.path.join(data_dir, "snapshot_*.csv.gz"))
    logging.debug(f"Found {len(snapshot_files)} snapshot files")
    
    if min_snapshots is None:
        min_snapshots = max(1, len(snapshot_files) // 2)  # Must appear in at least half the snapshots
    
    # Track coin occurrences and their maximum TVL across snapshots
    coin_counter = Counter()
    coin_max_tvl = defaultdict(float)  # Track maximum TVL for each coin
    coins_per_snapshot = []
    
    for snapshot_file in snapshot_files:
        logging.debug(f"Processing {os.path.basename(snapshot_file)}...")
        df = pd.read_csv(snapshot_file, compression="gzip")
        
        # Filter out rows with invalid token data (NaN symbols or IDs)
        df_clean = df.dropna(subset=['token0_id', 'token0_symbol', 'token1_id', 'token1_symbol', 'totalValueLockedUSD'])
        
        # Group by pool and get the maximum TVL for each pool in this snapshot
        df_clean['totalValueLockedUSD'] = pd.to_numeric(df_clean['totalValueLockedUSD'], errors='coerce')
        df_clean = df_clean.dropna(subset=['totalValueLockedUSD'])
        
        # Get all unique coins in this snapshot with their TVL values
        token0_coins = list(zip(df_clean['token0_id'], df_clean['token0_symbol'], df_clean['totalValueLockedUSD']))
        token1_coins = list(zip(df_clean['token1_id'], df_clean['token1_symbol'], df_clean['totalValueLockedUSD']))
        all_coins_with_tvl = token0_coins + token1_coins
        
        # Filter out coins with invalid symbols (ensure symbol is string)
        all_coins_with_tvl = [(coin_id, coin_symbol, tvl) for coin_id, coin_symbol, tvl in all_coins_with_tvl 
                             if isinstance(coin_symbol, str) and coin_symbol.strip() != '']
        
        # Get unique coins for this snapshot (remove duplicates, keep max TVL)
        coin_tvl_in_snapshot = {}
        for coin_id, coin_symbol, tvl in all_coins_with_tvl:
            coin_key = (coin_id, coin_symbol)
            if coin_key not in coin_tvl_in_snapshot or coin_tvl_in_snapshot[coin_key] < tvl:
                coin_tvl_in_snapshot[coin_key] = tvl
        
        snapshot_coins = set(coin_tvl_in_snapshot.keys())
        coins_per_snapshot.append(snapshot_coins)
        
        # Count occurrences and track maximum TVL across all snapshots
        for coin_key, tvl in coin_tvl_in_snapshot.items():
            coin_counter[coin_key] += 1
            coin_max_tvl[coin_key] = max(coin_max_tvl[coin_key], tvl)
    
    # Find coins that appear in at least min_snapshots
    qualifying_coins = []
    for coin_key, count in coin_counter.items():
        if count >= min_snapshots:
            max_tvl = coin_max_tvl[coin_key]
            qualifying_coins.append((coin_key, count, max_tvl))
    
    # Sort by TVL (descending) first, then by frequency (descending), then by symbol for consistency
    qualifying_coins.sort(key=lambda x: (-x[2], -x[1], x[0][1]))
    
    # Take the top max_coins with highest TVL
    common_coins = [coin_info[0] for coin_info in qualifying_coins[:max_coins]]
    
    logging.debug(f"\nFound {len(common_coins)} highest TVL coins (appearing in at least {min_snapshots} snapshots)")
    
    # Display TVL and frequency information
    logging.debug("Top coins by TVL and frequency:")
    for i, (coin_key, count, max_tvl) in enumerate(qualifying_coins[:max_coins]):
        coin_id, coin_symbol = coin_key
        logging.debug(f"  {i+1:2d}. {coin_symbol} ({coin_id[:10]}...): ${max_tvl:,.2f} TVL, {count}/{len(snapshot_files)} snapshots")
    
    return common_coins, snapshot_files