import networkx as nx
import math
import pandas as pd
import networkx as nx
from ast import literal_eval
import time 
import pandas as pd
from src.uniprice_2 import UniPrice

import argparse
import random
from src.algo_per_coin import safe_literal_eval
from src import uniswap
from src.graph import create_graph



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Uniswap V3 Crawler Benchmark")
    parser.add_argument("--csv-path", '-p', type=str, required=True, help="Path to the input CSV file")
    parser.add_argument("--top-token-counts", '--tc', type=int, default=5000, help="Number of top tokens to keep")
    args = parser.parse_args()

    time_prepare_data = time.time()
    df = pd.read_csv(args.csv_path)
        
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
    df[['token0_reserve', 'token1_reserve']] = [None, None]
       
    df_top_tokens = uniswap.get_top_tokens(df, args.top_token_counts)        
    
    # Select only the top tokens
    
    G = create_graph(df, apply_fee=False, remove_low_degree_nodes=False, remove_multi_edges=True, fake_fee=0)
    G_sub = G.subgraph(df_top_tokens['token_id']).copy()

    # df = df[df['token0_id'].isin(df_top_tokens['token_id']) & df['token1_id'].isin(df_top_tokens['token_id'])]
    # G_2 = create_graph(df, apply_fee=False, remove_low_degree_nodes=False, remove_multi_edges=True, fake_fee=0)
    
    # if

    time_prepare_data = time.time() - time_prepare_data
    print(f"Graph with {len(G_sub.nodes())} nodes and {len(G_sub.edges())} edges created in {time_prepare_data:.2f} seconds.")

    time_accept = time.time()

    uniprice = UniPrice()
    ans = uniprice.accept_graph(G_sub)
    
    time_accept = time.time() - time_accept    
    
    print(f"Time to accept graph: {time_accept:.2f} seconds")
    
    random.seed(22816000)
    coin_list = random.sample(df_top_tokens.head(args.top_token_counts)['token_id'].tolist(), 100)
            
    query_times = []
    for token_id in coin_list:
        try:
            query_time_start = time.time()
            uniprice.query_sssp(token_id)
            total_query_time = (time.time() - query_time_start)
            query_times.append(total_query_time)
        except Exception as e:
            pass
        # print(f"Query time for {token_id}: {elapsed:.4f} seconds")

    if query_times:
        avg_query_time = (sum(query_times) / len(query_times)) + (total_query_time/len(query_times))
        print(f"Average query time: {avg_query_time:.4f} seconds")