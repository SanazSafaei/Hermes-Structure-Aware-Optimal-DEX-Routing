import os
import time
import signal
import pandas as pd
import networkx as nx

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
import concurrent.futures
from multiprocessing import Pool, cpu_count
from datetime import datetime



def safe_literal_eval(val):
    try:
        if type(val) is dict:
            return val
        
        return literal_eval(val)
    except Exception:
        return None
    
def draw_labeled_multigraph(G, attr_name, ax=None):
    """
    Length of connectionstyle must be at least that of a maximum number of edges
    between pair of nodes. This number is maximum one-sided connections
    for directed graph and maximum total connections for undirected graph.
    """
    # Works with arc3 and angle3 connectionstyles
    connectionstyle = [f"arc3,rad={r}" for r in it.accumulate([0.15] * 4)]
    # connectionstyle = [f"angle3,angleA={r}" for r in it.accumulate([30] * 4)]

    pos = nx.shell_layout(G)
    nx.draw_networkx_nodes(G, pos, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=20, ax=ax)
    nx.draw_networkx_edges(
        G, pos, edge_color="grey", connectionstyle=connectionstyle, ax=ax
    )

    labels = {
        tuple(edge): f"{attr_name}={attrs[attr_name]}"
        for *edge, attrs in G.edges(keys=True, data=True)
    }
    nx.draw_networkx_edge_labels(
        G,
        pos,
        labels,
        connectionstyle=connectionstyle,
        label_pos=0.3,
        font_color="blue",
        bbox={"alpha": 0},
        ax=ax,
    )

def compute_snapshot_stats(block_number, G, df_top_tokens):

    

    stats_list = []
    jump_size = 5000
    # for top_coins in range(jump_size, 100000+jump_size, jump_size):
    for top_coins in [100, 1000]:
        # print('top_coins', top_coins)
        G_sub = G.subgraph(df_top_tokens.head(top_coins)['token_id'].to_list())

        stats_list.append({
                "block_number": block_number,
                "top_coins": top_coins,
                "num_nodes": len(G_sub.nodes),
                "num_edges": len(G_sub.edges),
                "max_degree": max(deg for _, deg in G_sub.degree()) if len(G_sub.nodes) > 0 else 0,
                "mean_degree": round(sum(deg for _, deg in G_sub.degree()) / len(G_sub.nodes), 2) if len(G_sub.nodes) > 0 else 0,
                "treewidth": nx.approximation.treewidth_min_degree(nx.Graph(G_sub.to_undirected()))[0] if len(G_sub.nodes) > 0 else 0,
            })

        stats_list.append({
                "block_number": block_number,
                "top_coins": top_coins,
                "num_nodes": None,
                "num_edges": None,
                "max_degree": None,
                "mean_degree": None,
                "treewidth": nx.approximation.treewidth_min_degree(nx.Graph(G_sub.to_undirected()))[0] if len(G_sub.nodes) > 0 else 0,
            })
        

    return pd.DataFrame(stats_list)


def process_snapshot(start_block_number, end_block_number, config):
    try:
        df_top_tokens = pd.read_csv(os.path.join(config['paths']['data'], 'top_tokens_22816000.csv'))
        logging.info(f"PID {os.getpid()} | Processing snapshot from block {start_block_number} to {end_block_number}")
        df_stats = pd.DataFrame()
        start_time = time.time()
        iter_count = 0
        total_blocks = end_block_number - start_block_number + 1

        for block_number, df, G in get_graph_from_to(
                config,
                start_block_number,
                end_block_number,
                top_token_list=df_top_tokens['token_id'].head(100000).tolist()):
            
            iter_count += 1
            df_res = compute_snapshot_stats(block_number, G, df_top_tokens)
            df_stats = pd.concat([df_stats, df_res], ignore_index=True)
            
            if iter_count % 5 == 0:
                elapsed = time.time() - start_time
                avg_time_per_iter = elapsed / iter_count if iter_count > 0 else 0
                remaining_iters = total_blocks - iter_count
                eta_seconds = avg_time_per_iter * remaining_iters
                eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))
                progress = (block_number - start_block_number + 1) / total_blocks * 100
                logging.info(
                    f"[PID {os.getpid()}] Block {block_number}: {progress:.2f}% complete "
                    f"({block_number - start_block_number + 1} of {total_blocks} blocks) | "
                    f"Elapsed: {elapsed:.1f}s | ETA: {eta_str}"
                )
        return df_stats

    except Exception as e:
        logging.error(f"Error processing snapshot for block {block_number}: {e}")
        return pd.DataFrame()


def aggregate_stats(start_block_number, end_block_number, max_workers, config):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    stats_dir = os.path.join(config['paths']['data'], 'stats')
    os.makedirs(stats_dir, exist_ok=True)
    stats_path = os.path.join(stats_dir, f'stats_{timestamp}.csv')

    header_written = False

    # Compute batch size based on the size of input (number of blocks)
    num_blocks = end_block_number - start_block_number
    # For example, target ~10 batches, but at least 1 block per batch
    # Make number of batches equal to max_workers (or cpu_count if not specified)
    num_batches = max_workers if max_workers else cpu_count()
    batch_size = max(1, (num_blocks + num_batches - 1) // num_batches)  # ceil division
    total_batches = num_batches

    with open(stats_path, 'w') as f:
        with Pool(num_batches) as pool:
            batch_ranges = [
                (start_block_number + i * batch_size,
                 min(start_block_number + (i + 1) * batch_size, end_block_number),
                 config)
                for i in range(total_batches)
            ]
            results = pool.starmap(process_snapshot, batch_ranges)
            for idx, df_stats in enumerate(results):
                if not df_stats.empty:
                    df_stats.to_csv(f, header=not header_written, index=False)
                    header_written = True
