import os
import time
import signal
import pandas as pd
import networkx as nx

from src.stats import safe_literal_eval
from src.update_graphs import get_graph_from_to
from .graph import create_graph
import logging
import tempfile
import sys
import networkx as nx
import matplotlib.pyplot as plt
import random
import math
import pandas as pd
import networkx as nx
import subprocess
import itertools as it
from ast import literal_eval
from src import graph, uniswap, sssp, uniprice, uniprice_2
import warnings
import multiprocessing
import concurrent.futures

from src.uniprice_2 import UniPrice
import multiprocessing
import signal
import math
import traceback
import concurrent.futures

class TimeoutError(Exception):
    """Custom timeout exception"""
    pass


def timeout_handler(signum, frame):
    """Signal handler for timeout"""
    raise TimeoutError("Algorithm execution timed out")
            
def create_graph_from_snapshot(config, block_number, top_coins_count_list):
    try:
        # Save results to CSV file
        results_dir = os.path.join(config['paths']['data'], 'results')
        os.makedirs(results_dir, exist_ok=True)
        
        snapshot_path = os.path.join(config['paths']['data'], 'snapshots', f'snapshot_{block_number}.csv.gz')
        
        # Read snapshot data
        df = pd.read_csv(snapshot_path)
        
        df['other'] = df['other'].apply(safe_literal_eval)
        df['token0Price'] = pd.to_numeric(df['token0Price'], errors='coerce')
        df['token1Price'] = pd.to_numeric(df['token1Price'], errors='coerce')
        df['totalValueLockedUSD'] = pd.to_numeric(df['totalValueLockedUSD'], errors='coerce')

        df = df.dropna(subset=['other', 'token0Price', 'token1Price'])

        # Check if 'feeTier' column exists, if not, create it
        if 'feeTier' not in df.columns:
            df['feeTier'] = df.apply(
                lambda row: int(row['other'].get('feeTier', 0)) if row['version'] in ['v3', 'v4'] 
                else 3000 if row['version'] == 'v2' 
                else None, 
                axis=1
            )
            df['feeTier'] = df['feeTier'].apply(float)
        
        #TODO here
                
        # TODO: Make this faster
        df_top_tokens = uniswap.get_top_tokens(df, max(top_coins_count_list))
        # Randomly select coins from the top tokens
            
        # Optimize filtering by converting lists to sets for faster lookup
        # token_set = set(df_top_tokens['token_id'].tolist() + coin_list)
        # df = df[df['token0_id'].isin(token_set) & df['token1_id'].isin(token_set)]

        # Calculate reserves for Uniswap
        df[['token0_reserve', 'token1_reserve']] = [None, None]
        # df = uniswap.add_reserves(df)
        

        G = create_graph(df, apply_fee=False, remove_low_degree_nodes=False, remove_multi_edges=True, fake_fee=0)
        # G = G.subgraph(df_top_tokens['token_id'].to_list()+coin_list)
        return G, df_top_tokens
    except Exception as e:
        logging.exception(e)
        return None
    


uniprice_dict = {}


def process_block_chunk(block_chunk, config, df_top_tokens, algorithm_name, top_coins_count, query_count, force, start_block_number):
    results_dir = os.path.join(config['paths']['data'], 'results')
    # os.makedirs(results_dir, exist_ok=True)
        
    uniprice = None
    if "ours" == algorithm_name:
        uniprice = UniPrice()

    logging.info(f"Processing block chunk {block_chunk[0]}-{block_chunk[-1]} | Algo: {algorithm_name} | Top: {top_coins_count} | PID {os.getpid()}")
    
    # if 22820545 >= block_chunk[0] and 22820545 <= block_chunk[-1]:
    #     print(f"Found block 22820545 in chunk {block_chunk[0]}-{block_chunk[-1]}, it's a special case with no data.", os.path.exists(os.path.join(results_dir, f"result_{algorithm_name}_22820545_{top_coins_count}.csv")))

    for block_number, df, G in get_graph_from_to(
            config,
            block_chunk[0],
            block_chunk[-1]+1,
            top_token_list=df_top_tokens['token_id'].tolist()
        ):
        start_time = time.time()
        # logging.info(f"######## Iter {block_number-start_block_number}: Start ########")
        result_csv_filename = f"result_{algorithm_name}_{block_number}_{top_coins_count}.csv"
        result_csv_path = os.path.join(results_dir, result_csv_filename)
    
        if os.path.exists(result_csv_path) and not force:
            # logging.info(f"File exists: {result_csv_path}. Skipping.")
            continue
        
        # print(f"Processing block {block_number} with {algorithm_name} algorithm...")
        
        df_res = run_algorithm_on_snapshot(
            algorithm_name, block_number, top_coins_count, query_count, force, config, G, df_top_tokens, uniprice
        )
        
        # This is just for a nice print!
        if len(df_res) > 0 and 'algo_time' in df_res.columns and 'prep_time' in df_res.columns:
            avg_time = (df_res['algo_time'].sum() + df_res['prep_time'].iloc[0]) / len(df_res)
            err_num = sum(df_res['err'].notna() & ~df_res['err'].str.lower().str.contains('timeout', na=False))
            timeout_num = sum(df_res['err'].str.lower().str.contains('timeout', na=False))
            output = (
            f"Block {block_number}: {algorithm_name} "
            f"{top_coins_count}: {avg_time:.4f}s, errors: {err_num}, timeouts: {timeout_num} | "
            f"time: {time.time() - start_time:.2f}s"
            )
            print(output)
        else:
            print(f"Block {block_number}: {algorithm_name} {top_coins_count}: No results | "
                  f"time: {time.time() - start_time:.2f}s")

def run_algorithm_from_to(
    algorithm_name,
    start_block_number,
    end_block_number,
    top_coins_count, 
    query_count,
    force,
    max_workers,
    config
):

    # Only read the first top_coins_count lines from the CSV
    df_top_tokens = pd.read_csv(
        os.path.join(config['paths']['data'], 'top_tokens_22816000.csv'),
        nrows=top_coins_count
    )

    # cpu_count = max(1, math.ceil(multiprocessing.cpu_count() / 2))
    # cpu_count = max(1, math.ceil(multiprocessing.cpu_count()-1))
    cpu_count = max(1, max_workers) if max_workers is not None else 24
    block_range = list(range(start_block_number, end_block_number + 1))
    chunk_size = math.ceil(len(block_range) / cpu_count)
    block_chunks = [block_range[i:i + chunk_size] for i in range(0, len(block_range), chunk_size)]

    # Prepare arguments for starmap
    pool_args = [
        (block_chunk, config, df_top_tokens, algorithm_name, top_coins_count, query_count, force, start_block_number)
        for block_chunk in block_chunks
    ]
    print(f"🦄 Buckle up! Processing {end_block_number-start_block_number} block chunks with {cpu_count} processes. Estimated time to finish: {(((end_block_number-start_block_number)*1300)/(3600*24))/cpu_count:.2f} days... Hope you brought snacks (or meals); this could take a while! 😱")

    with multiprocessing.Pool(cpu_count) as pool:
        pool.starmap(process_block_chunk, pool_args)




def run_single_query(coin_id, G_sub, algorithm_name, uniprice):


    algorithm_execution_time = time.time()
    try:
        distances = None
        # time.sleep(13)
        if algorithm_name == 'bf':
            nx.bellman_ford_predecessor_and_distance(G_sub, coin_id)
            # sssp.bellman_ford(G_sub, coin_id)
        elif algorithm_name == 'mmbf':
            sssp.modified_moore_bellman_ford(G_sub, coin_id)
        elif algorithm_name == 'bf2':
            raise NotImplementedError()
        elif algorithm_name == 'ours':
            # TODO: Add our algorithm again later, now mmbf and bf is more important!
            # raise NotImplementedError()
            uniprice.query_sssp(coin_id)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm_name}")

        return {
            'coin': coin_id,
            'err': None,
            'algo_time': time.time() - algorithm_execution_time,
            'algo': algorithm_name,
            'distances': distances
        }

    except Exception as e:
        pass
        # Only log exception if it's not a negative cycle detection
        if "negative" not in str(e).lower() and not isinstance(e, TimeoutError):
            logging.exception(f"Alg {algorithm_name}({coin_id}): Algorithm execution error: {e}")

        error_type = "timeout" if isinstance(e, TimeoutError) else "exception"
        return {
            'coin': coin_id,
            'err': f"{error_type}: {str(e)}",
            'algo_time': time.time() - algorithm_execution_time,
            'algo': algorithm_name,
            'distances': None
        }
            
def run_algorithm_on_snapshot(
    algorithm_name, 
    block_number, 
    top_coins_count, 
    query_count,
    force,
    config,
    G,
    df_top_tokens,
    uniprice
    ):
    """
    Execute an algorithm on a graph created from a snapshot and store the results.
    
    Args:
        algorithm_name (str): Name of the algorithm to execute
        coin_id (str): ID of the coin to use as source
        block_id (int): Block ID for the snapshot
        data_dir (str): Directory containing the data folder
    
    Returns:
        dict: Results dictionary with execution stats
    """
    
    try:

        results_dir = os.path.join(config['paths']['data'], 'results')
        os.makedirs(results_dir, exist_ok=True)
        
        if G is None:
            return None
        
        coin_list = []
        tasks = []
                  
        result_csv_filename = f"result_{algorithm_name}_{block_number}_{top_coins_count}.csv"
        result_csv_path = os.path.join(results_dir, result_csv_filename)
        
        # if block_number == 22820545:
        #     print("Found block 22820545, it's a special case with no data.", os.path.exists(result_csv_path), result_csv_path)

        if os.path.exists(result_csv_path) and not force:
            # logging.info(f"File exists: {result_csv_path}. Skipping.")
            return pd.DataFrame()

        # Initialize the subgraph and preprocessing time if not already done

        # print(f"Running {algorithm_name} on block {block_number} with top {top_coins_count} coins and query count {query_count}...")
    
        random.seed(block_number)
        coin_list = random.sample(df_top_tokens['token_id'].tolist(), query_count)
     
        # Calculate the preprocessing time
        preprocessing_time = time.time()
        
        if 'ours' == algorithm_name:
            uniprice.accept_graph(G)
            
        preprocessing_time = time.time() - preprocessing_time
        
        for coin_id in coin_list:
            tasks.append((coin_id, G, algorithm_name, uniprice))

        results = []

        # Sequential execution with 12 seconds timeout per task, Less process creation overhead!
        for args in tasks:
            try:
                # Set alarm for timeout
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(12)
                result = run_single_query(*args)
                signal.alarm(0)  # Disable alarm
                results.append(result)
            except TimeoutError as error:
                results.append({
                    'coin': args[0],
                    'err': f"timeout: {str(error)}",
                    'algo_time': 12,  # Timeout is 12 seconds
                    'algo': args[2],
                    'distances': None
                })
            except Exception as e:
                results.append({
                    'coin': args[0],
                    'err': f"exception: {str(e)}\n{traceback.format_exc()}",
                    'algo_time': None,
                    'algo': args[2],
                    'distances': None
                })
            finally:
                signal.alarm(0)  # Ensure alarm is always disabled~
             
        df_res = pd.DataFrame(results)
        if len(df_res) > 0:       
            df_res['block'] = block_number
            df_res['prep_time'] = preprocessing_time
            df_res['n_nodes'] = len(G.nodes)
            df_res['n_edges'] = len(G.edges)
            df_res['top_coin_count'] = top_coins_count
            df_res.to_csv(result_csv_path, index=False)
            
        return df_res

    except Exception as e:
        logging.exception(e)