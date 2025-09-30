from collections import defaultdict
import networkx as nx
from networkx import DiGraph
from ast import literal_eval
import math
import pandas as pd
import itertools
import logging
# set logging level to INFO
# logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
import time

import random
random.seed(42)


PLUS_INF_WEIGHT = 10**10 * 1.0
MINUS_INF_WEIGHT = -10**10 * 1.0


class UniPrice: 
    def __init__(self):
        
        # stores all edges in the graph: E + E^+
        self._all_edges_set = set()

        # stores the directed graph as networkx DiGraph
        self._graph = DiGraph()
        self._graph_filled = DiGraph()
        self._order = []

        # bookkeeping information
        self._fill_in_edge_to_parent_edge_pair = {}

        # stores the pre-singular vertices and singular vertices for identifying the arbitrage loops
        self._pre_singular_vertices = set()
        self._singular_vertices = set()



    def _need_recompute_fill_in_order(self):
        """Checks if we need to recompute the fill-in order based on the new pool"""
        # if there is an edge which is not in the _all_edges_set, we need to recompute the fill-in order
        for u, v in self._graph.edges():
            if (u, v) not in self._all_edges_set:
                # print(f"Need to recompute fill-in order: edge ({u}, {v}) not in _all_edges_set")
                return True
        return False

    # the following method is time consuming (the problem is NP-hard, but we can use tree decomposition as a good heuristic)
    def _recompute_fill_in_order(self):
        """returns the optimal fill-in order for the graph"""
        graph_copy = nx.Graph(self._graph)

        _treewidth, decomp_tree = nx.approximation.treewidth_min_degree(graph_copy)

        logging.info(f"Treewidth: {_treewidth}")
        # print(f"Decomposition tree: {decomp_tree}")

        bags = decomp_tree.nodes()
        triangulated_graph = graph_copy.copy()
        for bag in bags:
            for u, v in itertools.combinations(bag, 2):
                if not triangulated_graph.has_edge(u, v):
                    triangulated_graph.add_edge(u, v)

        # iterate over the triangulated graph, save the edges to the self._all_edges_set
        for u, v in triangulated_graph.edges():
            self._all_edges_set.add((u, v))
            self._all_edges_set.add((v, u))  

        peo = []
        seen_vertices = set()
        start_bag = list(bags)[0]
        post_ordered_bags = nx.dfs_postorder_nodes(decomp_tree, source=start_bag)
        for bag in post_ordered_bags:
            for vertex in bag:
                if vertex not in seen_vertices:
                    peo.append(vertex)
                    seen_vertices.add(vertex)

        self._order = peo

        num_graph_edges = len(self._graph.edges())
        if num_graph_edges > 0:
            logging.debug(f"Ratio new_edges/all_edges: {len(self._all_edges_set) / num_graph_edges}")
        else:
            logging.error("Ratio new_edges/all_edges: N/A (graph has no edges)")

        logging.debug(f"Ratio new_edges/all_edges: {len(self._all_edges_set) / len(self._graph.edges())}")




    def _backward_pass(self):
        """make the graph direct path compatible by filling in the edges according to the fill-in order"""
        self._graph_filled = self._graph.copy()
        
        self._fill_in_edge_to_parent_edge_pair.clear()
        self._pre_singular_vertices.clear()
        self._singular_vertices.clear() 

        # Iterate through vertices in reverse perfect elimination order
        for v in reversed(self._order):
            predecessors = list(self._graph_filled.predecessors(v))
            successors = list(self._graph_filled.successors(v))

            # Consider all paths of the form u -> v -> w
            for u in predecessors:
                for w in successors:
                    try:
                        weight_uv = self._graph_filled[u][v]['weight']
                        weight_vw = self._graph_filled[v][w]['weight']
                    except KeyError:
                        # This case should not be reached if graph weights are consistent
                        continue
                    
                    path_weight = weight_uv + weight_vw

                    # Case 1: u == w. This forms a 2-cycle u -> v -> u.
                    # Check if this cycle is a negative loop (arbitrage).
                    if u == w:
                        if path_weight < 0:
                            # Mark vertex `u` (which is part of the cycle) as pre-singular.
                            self._pre_singular_vertices.add(u)
                    
                    # Case 2: u != w. Check for a shorter path from u to w.
                    else:
                        # Get the current weight of the direct edge (u, w).
                        # If the edge doesn't exist, its weight is effectively infinity.
                        current_weight_uw = self._graph_filled.get_edge_data(u, w, default={'weight': PLUS_INF_WEIGHT})['weight']

                        # If the path through v is shorter, update (relax) the edge (u, w).
                        if path_weight < current_weight_uw:
                            self._graph_filled.add_edge(u, w, weight=path_weight)
                            
                            # Keep track of how this new edge weight was formed for path reconstruction.
                            self._fill_in_edge_to_parent_edge_pair[(u, w)] = ((u, v), (v, w))


    def _compute_singular_vertices(self):
        """
        This method computes the singular vertices based on the filled graph.
        
        It finds all Strongly Connected Components (SCCs) of the graph. If an SCC
        contains at least one pre-singular vertex (a vertex identified as part
        of a negative cycle), then all vertices in that SCC are marked as singular.
        """
        # Ensure the set of singular vertices is clear before starting.
        self._singular_vertices.clear()

        # If there are no pre-singular vertices, there's nothing to do.
        if not self._pre_singular_vertices:
            return

        sccs = nx.strongly_connected_components(self._graph_filled)

        # Iterate through each component.
        for scc in sccs:
            if not scc.isdisjoint(self._pre_singular_vertices):
                self._singular_vertices.update(scc)


    def accept_graph(self, G, need_singular_vertices=False):
        """Accepts a pool, check if we need to do the preprocessing again"""

        self._graph = G

        if self._need_recompute_fill_in_order():
            self._recompute_fill_in_order()
        
        self._backward_pass()

        if need_singular_vertices:
            self._compute_singular_vertices()


        ans = {
            'singular_vertices': self._singular_vertices,
            'pre_singular_vertices': self._pre_singular_vertices,
            'elimination_order': self._order,
        }

        return ans

    def query_sssp(self, token0, need_parents=False):
        """
        Returns a vector of shortest path distances from token0 to all other tokens,
        and a dictionary to reconstruct the shortest paths.
        This implementation is based on the Min-path algorithm (Algorithm 2) from
        Planken et al. (2012), which operates on a Directionally Path-Consistent graph.

        Returns:
            tuple: A tuple containing:
                - D (dict): A dictionary mapping each node to its shortest distance from token0.
                - parents (dict): A dictionary mapping each node to its predecessor on the shortest path from token0.
        """
        # First, check if the source token exists in the filled graph.
        if token0 not in self._graph_filled:
            raise ValueError(f"Token {token0} not found in the graph.")

        # Initialize the distance and parent dictionaries.
        D = {node: PLUS_INF_WEIGHT for node in self._graph_filled.nodes()}
        D[token0] = 0.0
        
        parents = {}
        if need_parents:
            parents = {node: None for node in self._graph_filled.nodes()}
        

        # For efficient lookups, create a map from vertex name to its index in the PEO.
        vertex_to_index = {vertex: i for i, vertex in enumerate(self._order)}
        
        # Get the index of the source vertex in the ordering.
        s_index = vertex_to_index[token0]

        # --- First Pass: Downward Sweep ---
        # This pass iterates from the source vertex `s` down to the beginning of the ordering.
        # It computes shortest paths from s to vertices that appear before it in the ordering.
        for k_idx in range(s_index, -1, -1):
            k = self._order[k_idx]
            
            # Find paths s -> ... -> k -> j where j appears before k in the ordering.
            for j in self._graph_filled.successors(k):
                if j in vertex_to_index:
                    j_idx = vertex_to_index[j]
                    if j_idx < k_idx:
                        # Relaxation step
                        new_dist = D[k] + self._graph_filled[k][j]['weight']
                        if new_dist < D[j]:
                            D[j] = new_dist
                            if need_parents:
                                parents[j] = k  # Update the parent pointer

        # --- Second Pass: Upward Sweep ---
        # This pass iterates from the beginning of the ordering to the end.
        # It computes shortest paths from s to vertices that appear after it in the ordering.
        for k_idx in range(len(self._order)):
            k = self._order[k_idx]
            
            # Find paths s -> ... -> k -> j where j appears after k in the ordering.
            for j in self._graph_filled.successors(k):
                if j in vertex_to_index:
                    j_idx = vertex_to_index[j]
                    if j_idx > k_idx:
                        # Relaxation step
                        new_dist = D[k] + self._graph_filled[k][j]['weight']
                        if new_dist < D[j]:
                            D[j] = new_dist
                            if need_parents:
                                parents[j] = k  # Update the parent pointer
                            
        # if dist > PLUS_INF_WEIGHT / 2 we replace it with inf 
        for node in D:
            if D[node] > PLUS_INF_WEIGHT / 2:
                D[node] = PLUS_INF_WEIGHT
        return D, parents


    def query_path(self, token0, token1):
            """
            Queries the shortest path from token0 to token1 by first finding the
            path in the filled graph and then recursively expanding the fill-in edges.
            """
            ans = {
                'distances': PLUS_INF_WEIGHT,
                'path': [],
            }

            # Step 1: Call SSSP to get the distances and parents dictionary for the filled graph.
            D, parents = self.query_sssp(token0, need_parents=True)

            # Step 2: Check if a path exists. If not, return an empty list.
            if D.get(token1, PLUS_INF_WEIGHT) >= PLUS_INF_WEIGHT:
                ans = { 
                    'distances': PLUS_INF_WEIGHT,
                    'path': [],
                }
                return ans

            # Step 3: Reconstruct the path from token0 to token1 in the filled graph.
            # This path may contain "shortcut" edges that need to be expanded.
            filled_path_nodes = []
            curr = token1
            while curr is not None:
                filled_path_nodes.append(curr)
                curr = parents.get(curr)
            filled_path_nodes.reverse()

            # Basic validation of the reconstructed path.
            if not filled_path_nodes or filled_path_nodes[0] != token0:
                return {
                    'distances': PLUS_INF_WEIGHT,
                    'path': [],
                }

            # Step 4: Iteratively expand the shortcut edges to find the true path.
            # We use a stack to manage the edges to be processed.
            final_path = [token0]
            
            # Create a stack of edges from the filled path.
            work_stack = []
            for i in range(len(filled_path_nodes) - 1):
                work_stack.append((filled_path_nodes[i], filled_path_nodes[i+1]))
            
            # Process edges from start to end by reversing the stack.
            work_stack.reverse()

            while work_stack:
                u, v = work_stack.pop()

                # Check if the edge (u, v) is a fill-in/shortcut edge.
                if (u, v) in self._fill_in_edge_to_parent_edge_pair:
                    # If it's a shortcut, expand it into its two parent edges.
                    (edge1, edge2) = self._fill_in_edge_to_parent_edge_pair[(u, v)]
                    
                    # Push the expanded edges back onto the stack. The second edge is
                    # pushed first so that the first edge (from u) is processed next.
                    work_stack.append(edge2)
                    work_stack.append(edge1)
                else:
                    # If it's an original edge, its destination is the next node in our final path.
                    final_path.append(v)
                    

            ans = {
                'distances': D[token1],
                'path': final_path
            }        
            return ans
        



# this is for testing 
# it generates random graph with no negative cycles
def random_weighted_no_negative_cycles(n: int, p: float, positive_magnitude: float, jonsons_magnitude: float) -> nx.DiGraph:
    """
    Generates a weighted, directed random graph with no negative cycles.

    This function first creates a random directed graph using the G(n,p)
    Erdős-Rényi model. It then assigns random positive weights to all edges.
    Finally, it uses a Johnson-style re-weighting with a random potential
    vector to introduce negative weights without creating negative cycles.

    Args:
        n: The number of nodes.
        p: The probability for edge creation.
        positive_magnitude: The maximum value for the initial random positive
                            edge weights (U[0, positive_magnitude]).
        jonsons_magnitude: The magnitude for the random potential vector
                           (U[-jonsons_magnitude, jonsons_magnitude]).

    Returns:
        A NetworkX DiGraph with weighted edges and no negative cycles.
    """
    # Step 1: Create a random directed graph using fast_gnp_random_graph.
    # This creates the structure of the graph.
    G = nx.fast_gnp_random_graph(n, p, directed=True)

    # Step 2: Assign initial random positive weights to all edges.
    # A graph with only positive weights is guaranteed to have no negative cycles.
    for u, v in G.edges():
        weight = random.uniform(0, positive_magnitude)
        G.add_edge(u, v, weight=weight)

    # Step 3: Generate a random potential vector (h), one value for each node.
    # These potentials will be used to re-weight the graph.
    potentials = {i: random.uniform(-jonsons_magnitude, jonsons_magnitude) for i in range(n)}

    # Step 4: Apply the potentials to re-weight the edges.
    # The new weight w'(u,v) = w(u,v) + h(u) - h(v).
    # This transformation preserves the shortest paths (once transformed back)
    # and does not introduce or remove negative cycles.
    for u, v, data in G.edges(data=True):
        original_weight = data['weight']
        new_weight = original_weight + potentials[u] - potentials[v]
        # Update the edge's weight in the graph
        G[u][v]['weight'] = new_weight

    return G




if __name__ == "__main__":
    need_draw = False

    # Example usage
    n = 1000  # Number of nodes
    p = 0.0007  # Probability of edge creation
    positive_magnitude = 1.0  # Maximum positive weight
    jonsons_magnitude = 5.0  # Magnitude for potential vector

    

    # let's create the APSP matrix 

    up = UniPrice()

    for trial in range(1000):

        G = random_weighted_no_negative_cycles(n, p, positive_magnitude, jonsons_magnitude, random_seed=random.randint(0, 1000000000000))
        
        time_start = time.time()
        result = up.accept_graph(G)
        time_to_accept = time.time() - time_start
        
        logging.info(f"Precompute time: {time_to_accept:.4f} seconds")

        try: 
            distance_dict_of_dicts = nx.floyd_warshall(G, weight='weight')
            all_pairs_distances = {
                (u, v): dist if not math.isinf(dist) else PLUS_INF_WEIGHT
                for u, targets in distance_dict_of_dicts.items()
                for v, dist in targets.items()
            }

            # let's do the same for the up
            all_pairs_distances_up = {} 
            all_vertices = list(G.nodes())
            
            all_times = []
            for u in all_vertices:
                time_iter_start = time.time()
                D, _ = up.query_sssp(u)
                # iterate over key to value, and add
                for v, dist in D.items():
                    all_pairs_distances_up[(u, v)] = dist
                time_iter_end = time.time()
                all_times.append(time_iter_end - time_iter_start)

            logging.info(f"Average time per vertex for SSSP: {sum(all_times) / len(all_times):.4f} seconds")

            # compare the two dictionaries
            all_keys = set(all_pairs_distances.keys()).union(set(all_pairs_distances_up.keys()))
            for key in all_keys:
                dist1 = all_pairs_distances.get(key, PLUS_INF_WEIGHT)
                dist2 = all_pairs_distances_up.get(key, PLUS_INF_WEIGHT)
                if math.isclose(dist1, dist2, abs_tol=1e-9):
                    continue
                else:
                    # print(f"Discrepancy found for {key}: {dist1} vs {dist2}")
                    # print(f"Faulty graph saved to debug_graph_{trial}.gpickle")
                    nx.write_gpickle(G, f"debug_graph_{trial}.gpickle")

                    # save the graph to a file for debugging
                     
                    raise ValueError(f"Discrepancy found for {key}: {dist1} vs {dist2}")
        
                
                

        except nx.NetworkXUnbounded:
            logging.error("negative cycle detected, cannot compute all-pairs shortest paths.")

        print(f"Test {trial + 1} passed.")

        # let's find all-pairs shortest paths 

    def query_path(self, token0, token1):
            """
            Queries the shortest path from token0 to token1 by first finding the
            path in the filled graph and then recursively expanding the fill-in edges.
            """
            ans = {
                'distances': PLUS_INF_WEIGHT,
                'path': [],
            }

            # Step 1: Call SSSP to get the distances and parents dictionary for the filled graph.
            D, parents = self.query_sssp(token0, need_parents=True)

            # Step 2: Check if a path exists. If not, return an empty list.
            if D.get(token1, PLUS_INF_WEIGHT) >= PLUS_INF_WEIGHT:
                ans = { 
                    'distances': PLUS_INF_WEIGHT,
                    'path': [],
                }
                return ans

            # Step 3: Reconstruct the path from token0 to token1 in the filled graph.
            # This path may contain "shortcut" edges that need to be expanded.
            filled_path_nodes = []
            curr = token1
            while curr is not None:
                filled_path_nodes.append(curr)
                curr = parents.get(curr)
            filled_path_nodes.reverse()

            # Basic validation of the reconstructed path.
            if not filled_path_nodes or filled_path_nodes[0] != token0:
                return {
                    'distances': PLUS_INF_WEIGHT,
                    'path': [],
                }

            # Step 4: Iteratively expand the shortcut edges to find the true path.
            # We use a stack to manage the edges to be processed.
            final_path = [token0]
            
            # Create a stack of edges from the filled path.
            work_stack = []
            for i in range(len(filled_path_nodes) - 1):
                work_stack.append((filled_path_nodes[i], filled_path_nodes[i+1]))
            
            # Process edges from start to end by reversing the stack.
            work_stack.reverse()

            while work_stack:
                u, v = work_stack.pop()

                # Check if the edge (u, v) is a fill-in/shortcut edge.
                if (u, v) in self._fill_in_edge_to_parent_edge_pair:
                    # If it's a shortcut, expand it into its two parent edges.
                    (edge1, edge2) = self._fill_in_edge_to_parent_edge_pair[(u, v)]
                    
                    # Push the expanded edges back onto the stack. The second edge is
                    # pushed first so that the first edge (from u) is processed next.
                    work_stack.append(edge2)
                    work_stack.append(edge1)
                else:
                    # If it's an original edge, its destination is the next node in our final path.
                    final_path.append(v)
                    

            ans = {
                'distances': D[token1],
                'path': final_path
            }        
            return ans
        


