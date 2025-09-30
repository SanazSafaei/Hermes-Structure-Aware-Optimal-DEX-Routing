from ast import literal_eval
import math
import os
import pickle
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from src import uniswap, uniprice_2
from src.graph import create_graph, update_graph_edges
import json, logging
import numpy as np
import re
import time
from pandarallel import pandarallel

pandarallel.initialize(verbose=False)

def safe_literal_eval(val):
    try:
        if type(val) is dict:
            return val
        
        return literal_eval(val)
    except Exception:
        return None
    

def get_closest_snapshot(config, block_number):
    checkout_snapshots = []
    for file_name in os.listdir(os.path.join(config['paths']['data'], 'snapshots')):
        match = re.search(r'\d+', file_name)
        if match:
            bn = match.group()
            checkout_snapshots.append(int(bn))

    # Find the closest available graph block number less than target_block_number
    closest_block_number = max([b for b in checkout_snapshots if b <= block_number], default=None)
    return closest_block_number


def get_closest_graph(config, block_number):
    checkout_snapshots = []
    for file_name in os.listdir(os.path.join(config['paths']['data'], 'graphs')):
        match = re.search(r'\d+', file_name)
        if match:
            bn = match.group()
            checkout_snapshots.append(int(bn))

    # Find the closest available graph block number less than target_block_number
    closest_block_number = max([b for b in checkout_snapshots if b <= block_number], default=None)
    return closest_block_number

# def read_and_falten_block(config, closest_block_number):
    
#     df = df.drop_duplicates('id')

#     df['token0Price'] = pd.to_numeric(df['token0Price'], errors='coerce')
#     df['token1Price'] = pd.to_numeric(df['token1Price'], errors='coerce')
#     df['totalValueLockedUSD'] = pd.to_numeric(df['totalValueLockedUSD'], errors='coerce')
#     df['other'] = df['other'].parallel_apply(safe_literal_eval)

#     df = df.dropna(subset=['other', 'token0Price', 'token1Price'])

#     # Check if 'feeTier' column exists, if not, create it
#     if 'feeTier' not in df.columns:
#         df['feeTier'] = df.parallel_apply(
#             lambda row: int(row['other'].get('feeTier', 0)) if row['version'] in ['v3', 'v4'] 
#             else 3000 if row['version'] == 'v2' 
#             else None, 
#             axis=1
#         )

#     if 'reserve0' not in df.columns:
#         df['reserve0'] = df['other'].parallel_apply(lambda x: pd.to_numeric(x.get('reserve0', None)) if x is not None else None)

#     if 'reserve1' not in df.columns:
#         df['reserve1'] = df['other'].parallel_apply(lambda x: pd.to_numeric(x.get('reserve1', None)) if x is not None else None)

#     df.set_index('id', inplace=True)
#     return df

def load_updates(config, block_number, closest_block_number):

    # Read all update files until target_block_number
    df_updates = pd.DataFrame()

    for file_name in os.listdir(os.path.join(config['paths']['data'], 'updates')):
        match = re.search(r'\d+', file_name)
        if match:
            bn = int(match.group())
            if bn >= closest_block_number and bn <= block_number:
                df_updates = pd.concat([df_updates, pd.read_csv(os.path.join(config['paths']['data'], 'updates', f'updates_{bn}.csv.gz'))])

    if df_updates.empty:
        raise ValueError(f"No updates found between blocks {closest_block_number} and {block_number}")
    
    df_updates = df_updates.reset_index().drop(columns='index').sort_values('block_number')
    return df_updates

def get_graph_from_to(config, start_block_number, end_block_number, top_token_list=[]):

    closest_block_number = get_closest_snapshot(config, start_block_number)
    
    if closest_block_number is None:
        logging.warning(f"No valid snapshot found for block {start_block_number}.")
        return []

    df_updates = load_updates(config, end_block_number, closest_block_number)

    df = pd.read_csv(
        os.path.join(config['paths']['data'], 'snapshots', f'snapshot_{closest_block_number}.csv')
        )
    df = df.set_index('id')
    df = df[(df['token0_id'].isin(top_token_list)) & (df['token1_id'].isin(top_token_list))]

    # df = df[(df.token0_id.isin(top_token_list))&(df.token1_id.isin(top_token_list))]
    
    try:
        with open(os.path.join(config['paths']['data'], "graphs", f"graph_{closest_block_number}.pkl"), "rb") as f:
            G = pickle.load(f)
    except Exception as e:
        logging.error(f"Failed to load graph for block {closest_block_number}")
        G = create_graph(df, apply_fee=True, remove_low_degree_nodes=False, remove_multi_edges=True, fake_fee=0)
    G = G.subgraph(top_token_list).copy()


    for block_number in range(closest_block_number, end_block_number):
        # Get all updates for this specific block
        block_updates = df_updates[df_updates.block_number == block_number]
        
        for i, row in block_updates.iterrows():
            try:    
                need_graph_update = False
                pool_id = row['pool_id']
                token0 = safe_literal_eval(row['token0'])
                token1 = safe_literal_eval(row['token1'])

                # Check if token0 and token1 are in the top_token_list otherwise skip
                if token0['id'] not in top_token_list or token1['id'] not in top_token_list:
                    continue

                # Check if pool is new and add to df if needed
                if pool_id not in df.index and row['pool_status'] == 'new':
                    # Create new row with data from the update
                    new_row = {
                        'token0_id': token0['id'],
                        'token1_id': token1['id'],
                        'version': row['version'],
                        'feeTier': float(row.get('feeTier', 3000)) if row['version'] == 'v2' else float(row.get('feeTier', 0)),
                        'reserve0': 0.0, # will be filled in the next step
                        'reserve1': 0.0, # will be filled in the next step
                        'token0Price': 0.0, # will be filled in the next step
                        'token1Price': 0.0, # will be filled in the next step
                        'sqrtPriceX96': float(row.get('sqrtPrice', 0)),
                        'tick': float(row.get('tick', 0))
                    }
                    df.loc[pool_id] = new_row
                
                # Skip if pool still not in df after potential addition
                if pool_id not in df.index:
                    continue
                    
                
                
                if row['version'] == 'v2':
                    # Get token decimals
                    token0_decimals = int(token0['decimals'])
                    token1_decimals = int(token1['decimals'])
                    
                    if row['event'] == 'swap':
                        # Update reserves: add inputs, subtract outputs, keep in smallest units
                        df.at[pool_id, 'reserve0'] = df.loc[pool_id, 'reserve0'] + float(token0['amount0'])
                        df.at[pool_id, 'reserve1'] = df.loc[pool_id, 'reserve1'] + float(token1['amount1'])
                        
                        # Calculate prices: token0Price = token1/token0, adjusted for decimals
                        df.at[pool_id, 'token0Price'] = (df.loc[pool_id, 'reserve1'] / df.loc[pool_id, 'reserve0']) * (10 ** (token0_decimals - token1_decimals))
                        df.at[pool_id, 'token1Price'] = (df.loc[pool_id, 'reserve0'] / df.loc[pool_id, 'reserve1']) * (10 ** (token1_decimals - token0_decimals))
                        need_graph_update = True
                        
                    elif row['event'] == 'mint':
                        # Add liquidity to reserves, keep in smallest units
                        
                        df.at[pool_id, 'reserve0'] = df.loc[pool_id, 'reserve0'] + float(token0['amount0'])
                        df.at[pool_id, 'reserve1'] = df.loc[pool_id, 'reserve1'] + float(token1['amount1'])
                        
                        # # Update prices based on new reserves, adjusted for decimals
                        df.at[pool_id, 'token0Price'] = (df.loc[pool_id, 'reserve1'] / df.loc[pool_id, 'reserve0']) * (10 ** (token0_decimals - token1_decimals))
                        df.at[pool_id, 'token1Price'] = (df.loc[pool_id, 'reserve0'] / df.loc[pool_id, 'reserve1']) * (10 ** (token1_decimals - token0_decimals))
                        need_graph_update = True

                        
                    elif row['event'] == 'burn':
                        # Remove liquidity from reserves, keep in smallest units
                        df.at[pool_id, 'reserve0'] = df.loc[pool_id, 'reserve0'] - float(token0['amount0'])
                        df.at[pool_id, 'reserve1'] = df.loc[pool_id, 'reserve1'] - float(token1['amount1'])
                        
                        # # Update prices based on new reserves, adjusted for decimals
                        # Check for division by zero before calculating prices
                        reserve0 = df.loc[pool_id, 'reserve0']
                        reserve1 = df.loc[pool_id, 'reserve1']
                        
                        if reserve0 != 0 and reserve1 != 0:
                            df.at[pool_id, 'token0Price'] = (reserve1 / reserve0) * (10 ** (token0_decimals - token1_decimals))
                            df.at[pool_id, 'token1Price'] = (reserve0 / reserve1) * (10 ** (token1_decimals - token0_decimals))
                        else:
                            # Set prices to 0 or NaN when reserves are zero
                            df.at[pool_id, 'token0Price'] = 0.0
                            df.at[pool_id, 'token1Price'] = 0.0
                        need_graph_update = True

                    # else:
                    #     raise NotImplementedError(f"Event {row['event']} for version {row['version']} not implemented")
                        
                elif row['version'] == 'v3':
                    if row['event'] == 'swap':
                        # Update pool state with new price and tick
                        df.at[pool_id, 'sqrtPriceX96'] = float(row['sqrtPrice'])
                        df.at[pool_id, 'tick'] = float(row['tick'])
                        df.at[pool_id, 'feeTier'] = float(row['feeTier'])
                        
                        # Calculate prices: token0Price = (sqrtPriceX96 / 2^96)^2, adjust for decimals
                        df.at[pool_id, 'token0Price'] = token0['price']
                        df.at[pool_id, 'token1Price'] = token1['price']
                        need_graph_update = True
                        
                    # elif row['event'] == 'mint':
                    #     pass
                        
                    # elif row['event'] == 'burn':
                    #     pass                
                        
                        
                    # else:
                    #     raise NotImplementedError(f"Event {row['event']} for version {row['version']} not implemented")
                        
                elif row['version'] == 'v4':
                    if row['event'] == 'swap':
                        # Update pool state with new price and tick
                        df.at[pool_id, 'sqrtPriceX96'] = float(row['sqrtPrice'])
                        df.at[pool_id, 'tick'] = float(row['tick'])
                        
                        # Calculate prices: token0Price = (sqrtPriceX96 / 2^96)^2, adjust for decimals
                        df.at[pool_id, 'token0Price'] = token0['price']
                        df.at[pool_id, 'token1Price'] = token1['price']
                        need_graph_update = True
                    # elif row['event'] == 'modifyLiquidity':
                    #     # Prices unchanged as modifyLiquidity doesn't affect sqrtPriceX96 or tick
                    #     pass
                        
                    # else:
                    #     raise NotImplementedError(f"Event {row['event']} for version {row['version']} not implemented")
                
                if need_graph_update:
                    update_graph_edges(
                        G,
                        token0_id=token0['id'],
                        token1_id=token1['id'],
                        token0_price=df.loc[pool_id, 'token0Price'],
                        token1_price=df.loc[pool_id, 'token1Price'],
                        feeTier=df.at[pool_id, 'feeTier']
                    )

            except Exception as e:
                logging.error(f"Error processing block {block_number} for pool {pool_id}: {e}")
                continue
    

        if block_number >= start_block_number:
            yield block_number, df, G



def add_check_points(config, start_block, end_block):

    closest_snapshot = get_closest_snapshot(config, end_block)
    closest_graph = get_closest_graph(config, end_block)

    start_time = time.time()
    for block_number, df, G in get_graph_from_to(config, min(closest_snapshot, closest_graph), end_block, []):

        if block_number % 100 == 0:
            snapshot_filename = f"snapshot_{block_number}.csv"
            # Save snapshot file when block number ends with 00
            snapshots_directory = os.path.join(config['paths']['data'], 'snapshots')
            df = df.reset_index()

            df_path = os.path.join(snapshots_directory, snapshot_filename)
            if not os.path.exists(df_path):
                df.to_csv(df_path, index=False)
                print(f"Snapshot saved: {snapshot_filename}")

            graph_filename = f"graph_{block_number}.pkl"
            graph_path = os.path.join(config['paths']['data'], "graphs", graph_filename)
            if not os.path.exists(graph_path):
                with open(graph_path, "wb") as f:
                    pickle.dump(G, f)
                print(f"Graph saved: {block_number}")
            print(f"Checkpoint at block {block_number} took {time.time()-start_time:.2f} seconds")



