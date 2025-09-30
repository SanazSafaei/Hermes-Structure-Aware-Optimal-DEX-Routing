import networkx as nx
import matplotlib.pyplot as plt
from src.stats import safe_literal_eval
import pandas as pd
import sys
import glob
import os
import pickle

def update_stats(config):
    df_top_tokens = pd.read_csv(os.path.join(config['paths']['data'], 'top_tokens_22816000.csv'))

    # Prepare set of top tokens for fast lookup
    top_token_set = set(df_top_tokens['token_id'].head(100000).tolist())

    # Track visited tokens and pools for this top_coins group
    visited_tokens = set()
    visited_pools = set()

    # Read the graph and add its vertices' token_id to visited_tokens
    graph_path = os.path.join(config['paths']['data'], 'graphs', 'graph_22816000.pkl')
    if os.path.exists(graph_path):
        with open(graph_path, 'rb') as f:
            G = pickle.load(f)
        visited_tokens.update(G.nodes)

    results_rows = []

    # Counters
    update_price = 0
    new_token = 0
    new_pool = 0

    # For each update file in data/updates
    updates_dir = os.path.join(config['paths']['data'], 'updates')
    if not os.path.exists(updates_dir):
        print(f"Directory {updates_dir} does not exist.")
        return

    update_files = sorted(glob.glob(os.path.join(updates_dir, "*.csv.gz")))
    for update_file in update_files:
        df_updates = pd.read_csv(update_file)
        df_updates['token0'] = df_updates['token0'].apply(safe_literal_eval)
        df_updates['token1'] = df_updates['token1'].apply(safe_literal_eval)

        # Each file contains updates for 100 blocks.
        # So, we should group by block_number and process each block within the file.
        if 'block_number' in df_updates.columns:

            for block, group in df_updates.groupby('block_number'):
                block_update_price = 0
                block_new_token = 0
                block_new_pool = 0
                for idx, row in group.iterrows():
                    token0 = row['token0']['id']
                    token1 = row['token1']['id']
                    pool_id = row['pool_id']

                    token0_in = token0 in top_token_set
                    token1_in = token1 in top_token_set

                    # New pool logic (per block)
                    if row['pool_status'] == 'new' and pool_id not in visited_pools:
                        block_new_pool += 1
                        visited_pools.add(pool_id)

                    # New token logic (per block)
                    if not token0_in or not token1_in:
                        if not token0_in and token0 not in visited_tokens:
                            block_new_token += 1
                            visited_tokens.add(token0)
                        if not token1_in and token1 not in visited_tokens:
                            block_new_token += 1
                            visited_tokens.add(token1)
                    else:
                        # Both tokens are in top_token_set
                        block_update_price += 1

                # Create a row for this block_number and top_coins
                row_dict = {
                    'block_number': block,
                    'top_coins': 100000,
                    'update_price': block_update_price,
                    'new_token': block_new_token,
                    'new_pool': block_new_pool
                }
                results_rows.append(row_dict) 

        print(f"Processed {block} for top_coins={100000}")

    df_results = pd.DataFrame(results_rows)
    df_results.to_csv(os.path.join(config['paths']['data'], 'update_stats.csv.gz'), index=False, compression='gzip')



