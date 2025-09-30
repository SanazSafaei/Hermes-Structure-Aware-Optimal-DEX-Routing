
import networkx as nx
import matplotlib.pyplot as plt
import random
import math
import pandas as pd
import networkx as nx
import subprocess
import itertools as it
from ast import literal_eval
from src import graph, uniswap

def bfs_distances(G, source):
    """
    Computes shortest paths and distances from the source node to all other nodes in a graph G using BFS.
    Assumes edge weights are stored in G[u][v]['weight'].
    Returns:
        bfs_paths: dict mapping node to path from source.
        distances: dict mapping node to distance from source.
    """
    bfs_paths = {source: [source]}
    distances = {source: 0}
    for parent, child in nx.bfs_edges(G, source=source):
        # Path to child is path to parent + [child]
        bfs_paths[child] = bfs_paths[parent] + [child]
        # Distance to child is distance to parent + edge weight
        distances[child] = distances[parent] + G[parent][child]['weight']

    return bfs_paths, distances 


def modified_moore_bellman_ford(G, source):
    """
    Computes shortest paths in a directed graph G from a source node using a modified Moore-Bellman-Ford algorithm.
    This version works on the line graph of G, as described in the referenced paper.
    Returns:
        P_token: dict mapping node to path (as list of edges) from source.
        D_token: dict mapping node to distance from source.
    """
    # Copy the graph and add a dummy origin node 'O' connected to the source
    G_cop = G.copy()
    G_cop.add_edge('O', source, weight=0)

    # Create the line graph L(G) where nodes are edges of G, and edges represent consecutive edges in G
    L_G = nx.DiGraph()
    for u, v in G_cop.edges():
        for w in G_cop.successors(v):
            # Edge from (u, v) to (v, w) with weight of (v, w)
            L_G.add_edge((u, v), (v, w), weight=G[v][w]['weight'])

    # Initialize distances and paths for all nodes in the line graph
    distances = {node: float('inf') for node in L_G.nodes()}
    paths = {node: [] for node in L_G.nodes()}

    # Distance to the starting edge is zero
    distances[('O', source)] = 0

    # Relax edges repeatedly (Bellman-Ford)
    for _ in range(len(G_cop.nodes) - 1):
        for (v_i, v_j), (v_j2, v_l) in L_G.edges():
            weight = L_G[(v_i, v_j)][(v_j2, v_l)]['weight']
            # Flatten the path to check for cycles
            #TODO: do it faster
            flat_nodes = [node for pair in paths[(v_i, v_j)] for node in pair]
            # Relaxation step with cycle check
            if distances[(v_i, v_j)] + weight < distances[(v_j2, v_l)]:
                if v_l not in flat_nodes or v_l == source:
                    distances[(v_j2, v_l)] = distances[(v_i, v_j)] + weight
                    paths[(v_j2, v_l)] = paths[(v_i, v_j)] + [(v_j2, v_l)]

    # Extract shortest distances and paths for each token (node in G)
    D_token = {node: float('inf') for node in G_cop.nodes()}
    P_token = {}
    for (v_i, v_j) in distances:
        v_d = distances[(v_i, v_j)]
        t = v_j
        if v_d < D_token[t]:
            D_token[t] = v_d
            P_token[t] = paths[(v_i, v_j)]
    return P_token, D_token  




def bellman_ford(G, Source):
    """
    Computes shortest paths from the Source node to all other nodes in a weighted directed graph G using the Bellman-Ford algorithm.
    Returns:
        paths: dict mapping node to path from Source.
        distances: dict mapping node to distance from Source.
    """
    # Initialize distances and predecessors
    distances = {node: float('inf') for node in G.nodes()}
    predecessors = {node: None for node in G.nodes()}
    distances[Source] = 0

    # Relax edges repeatedly
    for _ in range(len(G.nodes()) - 1):
        for u, v, data in G.edges(data=True):
            weight = data.get('weight', 1)
            if distances[u] + weight < distances[v]:
                distances[v] = distances[u] + weight
                predecessors[v] = u

    # Check for negative-weight cycles
    for u, v, data in G.edges(data=True):
        weight = data.get('weight', 1)
        if distances[u] + weight < distances[v]:
            raise ValueError("Graph contains a negative-weight cycle")

    # Build paths from Source to each node
    paths = {Source: [Source]}
    for node in G.nodes():
        if node == Source:
            continue
        path = []
        current = node
        while current is not None and current != Source:
            path.append(current)
            current = predecessors[current]
        if current == Source:
            path.append(Source)
            paths[node] = list(reversed(path))
        else:
            paths[node] = []

    return paths, distances