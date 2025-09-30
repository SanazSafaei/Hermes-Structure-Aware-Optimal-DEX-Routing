import time
import csv
import logging, random
import requests, os
from web3 import Web3
import pandas as pd
import src.query as query_strings
from src import api, update as up
import threading
from multiprocessing.pool import ThreadPool
import multiprocessing as mp
import numpy as np

def fetch_liquidity_pools(count, skip, the_graph_token, config, version = 'v3'):
    error_counter = 0
    dataframes = []

    v2_other_columns = [
        'totalSupply',
        'reserveETH',
        'reserveUSD',
        'trackedReserveETH',
        'reserve1',
        'reserve0'
    ]

    v3_other_columns = [
        'feeTier',
        'sqrtPrice',
        'tick',
        'feesUSD'
    ]

    # time.sleep(1)

    if (version == 'v2'):
        query = query_strings.pools_query_v2.replace("order_by_param", 'reserveUSD')
        other_columns = v2_other_columns
    else:
        query = query_strings.pools_query.replace("order_by_param", 'totalValueLockedUSD')
        other_columns = v3_other_columns

    query = query.replace("first_num", str(count)).replace("skip_num", str(skip))
    
    res = api.the_graph_query_wrapper(query, the_graph_token, config=config, version=version)

    
    if version == "v2":
        data = res['data']['pairs']
    else:
        data = res['data']['pools']
    

    df = pd.DataFrame(data)
    if version == 'v2':
        df['liquidity'] = np.sqrt(df['reserve0'].apply(float)*df['reserve1'].apply(float))
        df['totalValueLockedUSD'] = df['reserveUSD'] # Probably the same thing

    # Add an 'other' column with JSON objects for the specified columns
    df['other'] = df[other_columns].apply(lambda row: row.dropna().to_dict(), axis=1)

    # Drop the original columns that are now part of the 'other' column
    df = df.drop(columns=other_columns)

    # Dynamically expand all nested JSON objects and lists of JSON objects
    for column in df.columns:
        
        if column == 'other':
            continue

        if isinstance(df[column].iloc[0], dict):  # Check if the column contains JSON objects
            nested_df = pd.json_normalize(df[column])
            nested_df.columns = [f"{column}_{col}" for col in nested_df.columns]  # Prefix to avoid collisions
            df = pd.concat([df, nested_df], axis=1)
            df = df.drop(columns=[column])  # Drop the original column
        elif isinstance(df[column].iloc[0], list):  # Check if the column contains lists of JSON objects
            # Expand lists of JSON objects into separate rows
            expanded_rows = df[column].explode().reset_index(drop=True)
            nested_df = pd.json_normalize(expanded_rows)
            nested_df.columns = [f"{column}_{col}" for col in nested_df.columns]  # Prefix to avoid collisions
            df = pd.concat([df.drop(columns=[column]).reset_index(drop=True), nested_df], axis=1)
    
    df['version'] = version

    return df

class PoolFetcherThread(threading.Thread):
    def __init__(self, config, current_block):
        super().__init__()
        self.config = config
        self.the_graph_tokens = config['services']['the_graph_tokens']
        self.current_block = current_block
        self.df = pd.DataFrame()
        self.mp_pool = None

    def run(self):

        self.mp_pool = mp.Pool(processes=mp.cpu_count())         


        for skip in range(0, 500000, 1000):
            self.mp_pool.apply_async(fetch_liquidity_pools, args=(1000, skip, random.choice(self.the_graph_tokens), self.config, 'v2'), callback=self.callback)

        for skip in range(0, 15000, 1000):
            self.mp_pool.apply_async(fetch_liquidity_pools, args=(1000, skip, random.choice(self.the_graph_tokens), self.config, 'v3'), callback=self.callback)
        
        for skip in range(0, 4000, 1000):
            self.mp_pool.apply_async(fetch_liquidity_pools, args=(1000, skip, random.choice(self.the_graph_tokens), self.config, 'v4'), callback=self.callback)
        
        self.mp_pool.close()
        self.mp_pool.join()

        self.df['block_number'] = self.current_block

        logging.info(f"Fetched {len(self.df)} pools for block {self.current_block}")
        if not self.df.empty:
            logging.info(f": Versions: {str(dict(self.df.version.value_counts()))}" if 'version' in self.df else "")

    def callback(self, result):
        self.df = pd.concat([self.df, result], ignore_index=True)

    def stop(self):
        self.mp_pool.terminate()
        


# class SwapFetcherThread(threading.Thread):
#     def __init__(self, config, start_block):
#         super().__init__()
#         self.config = config
#         self.the_graph_tokens = config['services']['the_graph_tokens']
#         self.df = pd.DataFrame()
#         self.mp_pool = None
#         self.start_block = start_block
#     def run(self):
#
#         self.mp_pool = mp.Pool(processes=1)         
#
#         self.mp_pool.apply_async(up.fetch_changes_transactions_v3_v4, args=(1000, 0, random.choice(self.the_graph_tokens), self.start_block, self.config), callback=self.callback)
#             
#         self.mp_pool.close()
#         self.mp_pool.join()
#
#         if(len(self.df)>0):
#             logging.info(f"Fetched {len(self.df)} pools for block {self.start_block} version: {self.df['version'].values[0]}")
#
#     def callback(self, result):
#         self.df = pd.concat([self.df, result], ignore_index=True)
#
#
#     def stop(self):
#         self.mp_pool.terminate()
        

def update(config, start_block, end_block):
    """
    Use multiprocessing.Pool to fetch swap changes in parallel batches.
    """
    max_processes = config['download']['n_threads']
    
    # Multiprocessing version
    # Prepare arguments for each batch
    batch_args = []
    for batch_start in range(start_block, end_block, 100):
        # Use the new signature: (start_block, end_block, count, skip, config)
        # Here, count=1000, skip=0 as before
        batch_args.append((batch_start, min(batch_start + 100, end_block), 1000, 0, config))
    
    with mp.Pool(processes=max_processes) as pool:
        results = pool.starmap(up.fetch_changes_transactions, batch_args)

    # Single-threaded version
    # results = []
    # for batch_start in range(start_block, end_block, 100):
    #     batch_end = min(batch_start + 100, end_block)
    #     df = up.fetch_changes_transactions(batch_start, batch_end, 1000, 0, config)
    #     results.append(df)

    # Collect results
    # for batch_start, df in zip(range(start_block, end_block, 100), results):
    #     try:
    #         if df is not None and not df.empty:
    #             logging.info(f"Fetched {len(df)} pools for block {batch_start} version: {df['version'].values[0]}")
    #     except Exception as e:
    #         logging.error(f"Error fetching swaps for block {batch_start}: {e}")
        
    # logging.info(f"Fetched {len(results)} batches of pool updates from block {start_block} to {end_block}")

        
def download(config):
    """Monitor Ethereum blockchain for new blocks and query Uniswap pools."""

    current_block = 0
    fetcher_thread = None

    # TODO: For debug
    while True:
        try:
            web3 = Web3(Web3.HTTPProvider(random.choice(config['services']['ethereum_providers'])))
            # new_block_number = api.get_eth_block_number(random.choice(config['services']['ethereum_providers']))
            new_block_number = web3.eth.block_number
            
            if current_block < new_block_number:
                # idk why but threads after first thread are not alive!
                # SF: Who are you?
                if fetcher_thread:
                    fetcher_thread.stop()

                    if len(fetcher_thread.df) > 0:
                        try:
                            snapshots_directory = os.path.join(config['paths']['data'], 'snapshots')
                            os.makedirs(snapshots_directory, exist_ok=True)
                            fetcher_thread.df.to_csv(os.path.join(snapshots_directory, f'snapshot_{current_block}.csv.gz'), index=False, compression='gzip')
                        except Exception as e:
                            print(e)

                current_block = new_block_number
                logging.info(f"New block detected: {current_block}")

                fetcher_thread = PoolFetcherThread(config, current_block)
                # fetcher_thread = UpdateFetcherThread(config, current_block)
                fetcher_thread.start()

            time.sleep(0.1)
        except Exception as e:
            logging.error(f"Error in download loop: {e}")
        
        
        
def download_all(config):
    """Monitor Ethereum blockchain for new blocks and query Uniswap pools."""
    current_block = 0
    fetcher_thread = None

    try:
        web3 = Web3(Web3.HTTPProvider(random.choice(config['services']['ethereum_providers'])))
        current_block = web3.eth.block_number

        logging.info(f"New block detected: {current_block}")

        fetcher_thread = PoolFetcherThread(config, current_block)
        fetcher_thread.start()
        fetcher_thread.join()  # Wait for the thread to finish

        if len(fetcher_thread.df) > 0:
            try:
                snapshots_directory = os.path.join(config['paths']['data'], 'snapshots')
                os.makedirs(snapshots_directory, exist_ok=True)
                fetcher_thread.df.to_csv(os.path.join(snapshots_directory, f'snapshot_{current_block}.csv.gz'), index=False, compression='gzip')
            except Exception as e:
                print(e)

    except Exception as e:
        logging.error(f"Error in download_all: {e}")