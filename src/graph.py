import math
import pandas as pd
import networkx as nx
import subprocess
import itertools as it
from ast import literal_eval


def update_graph_edges(G, token0_id, token1_id, token0_price, token1_price, feeTier):
    
    # Apply fee to prices if feeTier is provided
    if feeTier is not None:
        token0_price = token0_price * (1 - feeTier / 1e6)
        token1_price = token1_price * (1 - feeTier / 1e6)

    # token1 -> token0
    weight_0 = math.log(token0_price) * -1 if float(token0_price) > 0 else 1e10
    if G.has_edge(token1_id, token0_id):
        existing_weight = G[token1_id][token0_id]['weight']
        G[token1_id][token0_id]['weight'] = min(existing_weight, weight_0)
    else:
        G.add_edge(token1_id, token0_id, weight=weight_0)

    # token0 -> token1
    weight_1 = math.log(token1_price) * -1 if float(token1_price) > 0 else 1e10
    if G.has_edge(token0_id, token1_id):
        existing_weight = G[token0_id][token1_id]['weight']
        G[token0_id][token1_id]['weight'] = min(existing_weight, weight_1)
    else:
        G.add_edge(token0_id, token1_id, weight=weight_1)
        
def create_graph(df, apply_fee=False, remove_low_degree_nodes=False, remove_multi_edges=False, delta_fees=None, fake_fee=0.0):
    # Create a graph
    if remove_multi_edges:
        G = nx.DiGraph()
    else:
        G = nx.MultiDiGraph()
        
    # df['other'] = df['other'].apply(safe_literal_eval)
    # df['token0Price'] = pd.to_numeric(df['token0Price'], errors='coerce')
    # df['token1Price'] = pd.to_numeric(df['token1Price'], errors='coerce')
    # df = df.dropna(subset=['other', 'token0Price', 'token1Price'])

    # # Check if 'feeTier' column exists, if not, create it
    # if 'feeTier' not in df.columns:
    #     df['feeTier'] = df.apply(
    #         lambda row: int(row['other'].get('feeTier', 0)) if row['version'] in ['v3', 'v4'] 
    #         else 3000 if row['version'] == 'v2' 
    #         else None, 
    #         axis=1
    #     )
    #     df['feeTier'] = df['feeTier'].apply(float)
    
    if apply_fee:
        df['token0Price'] = df['token0Price'] * (1 - df['feeTier'] / 1e6)
        df['token1Price'] = df['token1Price'] * (1 - df['feeTier'] / 1e6)

    if fake_fee > 0.0:
        df['token0Price'] = df['token0Price'] * (1 - fake_fee)
        df['token1Price'] = df['token1Price'] * (1 - fake_fee)

    for _, row in df.iterrows():
        # Add nodes
        G.add_node(row['token0_id'])
        G.add_node(row['token1_id'])

        # token1 -> token0
        weight_0 = math.log(row['token0Price']) * -1 if float(row['token0Price']) > 0 else 1e10
        if remove_multi_edges:
            if G.has_edge(row['token1_id'], row['token0_id']):
                existing_weight = G[row['token1_id']][row['token0_id']]['weight']
                if weight_0 < existing_weight:
                    G[row['token1_id']][row['token0_id']]['weight'] = weight_0
            else:
                G.add_edge(row['token1_id'], row['token0_id'], weight=weight_0, version=row.get('version', None))
        else:
            G.add_edge(row['token1_id'], row['token0_id'], weight=weight_0, version=row.get('version', None))

        # token0 -> token1
        weight_1 = math.log(row['token1Price']) * -1 if float(row['token1Price']) > 0 else 1e10
        if remove_multi_edges:
            if G.has_edge(row['token0_id'], row['token1_id']):
                existing_weight = G[row['token0_id']][row['token1_id']]['weight']
                if weight_1 < existing_weight:
                    G[row['token0_id']][row['token1_id']]['weight'] = weight_1
            else:
                G.add_edge(row['token0_id'], row['token1_id'], weight=weight_1, version=row.get('version', None))
        else:
            G.add_edge(row['token0_id'], row['token1_id'], weight=weight_1, version=row.get('version', None))
    
    # Remove nodes with degree less than 2 if the flag is set
    if remove_low_degree_nodes:
        while True:
            nodes_to_remove = [node for node in G.nodes() if G.in_degree(node) == 1 and G.out_degree(node) == 1]
            if len(nodes_to_remove) == 0:
                break
            G.remove_nodes_from(nodes_to_remove)
    return G
