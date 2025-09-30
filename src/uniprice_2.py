"""
This module implements the UniPrice algorithm for computing shortest paths in a directed graph.
"""

from collections import defaultdict
import math
import itertools
import logging
import time
import random

import networkx as nx
from networkx import DiGraph

# set logging level to INFO
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# for random seed
random.seed(42)

PLUS_INF_WEIGHT = 10**10 * 1.0
MINUS_INF_WEIGHT = -10**10 * 1.0


# Changed this to allow multiprocessing!
def default_plus_inf_weight():
    return PLUS_INF_WEIGHT

def default_nested_plus_inf_weight():
    return defaultdict(default_plus_inf_weight)

class UniPrice: 
    def __init__(self):
        
        # stores all edges in the graph: E + E^+
        self._all_edges_set = set()

        # stores the directed graph as networkx DiGraph
        self._graph = DiGraph()
        

        
        self._weights = defaultdict(default_nested_plus_inf_weight)
        self._successors = defaultdict(list)
        self._predecessors = defaultdict(list)
        self._order = []

        # bookkeeping information
        self._fill_in_edge_to_parent_edge_pair = {}

        # stores the pre-singular vertices and singular vertices for identifying the arbitrage loops
        self._pre_singular_vertices = set()
        self._singular_vertices = set()



    def _need_recompute_fill_in_order(self):
        """Checks if we need to recompute the fill-in order based on the new pool"""
        # if there is an edge which is not in the _all_edges_set, 
        # we need to recompute the fill-in order
        for u, v in self._graph.edges():
            if (u, v) not in self._all_edges_set:
                # print(f"Need to recompute fill-in order: edge ({u}, {v}) not in _all_edges_set")
                return True
        return False


    def _find_peo_from_clique_tree(self, decomp_tree: nx.Graph) -> list:
        """
        Finds a Perfect Elimination Ordering (PEO) from a tree decomposition using
        an efficient, single-pass algorithm based on a post-order traversal.

        Args:
            decomp_tree: A NetworkX graph representing the tree decomposition.
                        Each node in this graph is a tuple of vertices (a bag).

        Returns:
            A list of vertices representing the correct Perfect Elimination Ordering.
        """
        # --- Handle edge cases for empty or trivial graphs ---
        if not decomp_tree or decomp_tree.number_of_nodes() == 0:
            return []
        if decomp_tree.number_of_nodes() == 1:
            # If there's only one bag, the PEO is just its vertices.
            return list(list(decomp_tree.nodes())[0])

        # --- 1. Perform a single traversal to get post-order and parent links ---
        # Pick an arbitrary start node for the traversal.
        start_bag = next(iter(decomp_tree.nodes()))
        
        # Get the bags in post-order from our starting point.
        post_ordered_bags = list(nx.dfs_postorder_nodes(decomp_tree, source=start_bag))
        
        # Get the parent of each node in the DFS tree. The root has no parent.
        parents = nx.dfs_predecessors(decomp_tree, source=start_bag)

        peo = []
        processed_vertices = set()

        # --- 2. Process all non-root bags in post-order ---
        # The root of the DFS tree is the last element in the post-order list.
        # We iterate through all other bags first.
        for bag in post_ordered_bags[:-1]:
            # The parent is guaranteed to exist for any non-root node.
            parent_bag = parents[bag]
            
            # The vertices to eliminate are those in the current bag but not its parent.
            vertices_to_eliminate = set(bag) - set(parent_bag)

            for vertex in vertices_to_eliminate:
                if vertex not in processed_vertices:
                    peo.append(vertex)
                    processed_vertices.add(vertex)

        # --- 3. Handle the root bag ---
        # The root is the last bag in the post-order list.
        root_bag = post_ordered_bags[-1]
        for vertex in root_bag:
            if vertex not in processed_vertices:
                peo.append(vertex)
                processed_vertices.add(vertex)

        # The list `peo` is now the correct elimination order, e.g., [v1, v2, ..., vn].
        return list(reversed(peo))  # Reverse to get the order from first to last elimination.

    def _recompute_fill_in_order(self):
        """returns the optimal fill-in order for the graph"""
        graph_copy = nx.Graph(self._graph)
        _treewidth, decomp_tree = nx.approximation.treewidth_min_degree(graph_copy)

        logging.debug(f"Treewidth: {_treewidth}")
        logging.debug(f"Edges in the decomposition tree: {decomp_tree.edges()}")
        
        bags = decomp_tree.nodes()
        logging.debug(f"Bags in the decomposition tree: {bags}")

        # Triangulate the graph: for every bag, make it a clique.
        triangulated_graph = graph_copy.copy()
        for bag in bags:
            for u, v in itertools.combinations(bag, 2):
                if not triangulated_graph.has_edge(u, v):
                    triangulated_graph.add_edge(u, v)

        logging.debug(f"All edges in the triangulated graph: {triangulated_graph.edges()}")

        # iterate over the triangulated graph, save the edges to the self._all_edges_set
        self._all_edges_set.clear() # Clear previous edges
        for u, v in triangulated_graph.edges():
            self._all_edges_set.add((u, v))
            self._all_edges_set.add((v, u))  

        peo = self._find_peo_from_clique_tree(decomp_tree)
        # --- End of new, corrected logic ---

        self._order = peo

        logging.debug(f"Perfect elimination order: {self._order}")

        num_graph_edges = len(self._graph.edges())
        if num_graph_edges > 0:
            # Note: self._all_edges_set contains tuples, so its length is the number of unique edges.
            logging.debug(f"Fill-in ratio (new_edges / original_edges): {(len(self._all_edges_set) - num_graph_edges) / num_graph_edges:.2f}")
        else:
            logging.info("Fill-in ratio: N/A (original graph has no edges)")


    # the following method is time consuming (the problem is NP-hard, 
    # but we can use tree decomposition as a good heuristic)
    def _recompute_fill_in_order_err(self):
        """returns the optimal fill-in order for the graph"""
        graph_copy = nx.Graph(self._graph)
        _treewidth, decomp_tree = nx.approximation.treewidth_min_degree(graph_copy)

        logging.info(f"Treewidth: {_treewidth}")
        # print(f"Decomposition tree: {decomp_tree}")

        bags = decomp_tree.nodes()
        # pring edges in the decomposition tree
        logging.debug(f"Edges in the decomposition tree: {decomp_tree.edges()}")
        # print(f"Bags in the decomposition tree: {bags}")        

        triangulated_graph = graph_copy.copy()
        for bag in bags:
            for u, v in itertools.combinations(bag, 2):
                if not triangulated_graph.has_edge(u, v):
                    triangulated_graph.add_edge(u, v)


        # for debugging purposes, print the bags: 
        logging.debug(f"Bags in the decomposition tree: {bags}")
        logging.debug(f"All edges in the triangulated graph: {triangulated_graph.edges()}")

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


        logging.debug(f"Perfect elimination order: {self._order}")

        num_graph_edges = len(self._graph.edges())
        if num_graph_edges > 0:
            logging.debug(f"Ratio new_edges/all_edges: {len(self._all_edges_set) / num_graph_edges}")
        else:
            logging.error("Ratio new_edges/all_edges: N/A (graph has no edges)")

        logging.debug(f"Ratio new_edges/all_edges: {len(self._all_edges_set) / len(self._graph.edges())}")

    def _backward_pass(self):
        """
        make the graph direct path compatible by filling in the edges according to the fill-in order.
        This is a heavily optimized version. It populates and operates on native Python
        dictionaries and lists instead of networkx objects for maximum speed.
        """
        # --- OPTIMIZATION: Clear and populate fast data structures ---
        self._weights.clear()
        self._successors.clear()
        self._predecessors.clear()
        self._fill_in_edge_to_parent_edge_pair.clear()
        self._pre_singular_vertices.clear()
        self._singular_vertices.clear() 
        _complexity_counter = 0
        _new_edges_counter = 0

        # Populate initial weights and adjacency lists from the base graph
        nodes = self._graph.nodes()
        for u in nodes:
            # Ensure all nodes exist in our structures, even if they have no edges
            self._successors[u] = []
            self._predecessors[u] = []

        for u, v, data in self._graph.edges(data=True):
            weight = data.get('weight', PLUS_INF_WEIGHT)
            self._weights[u][v] = weight
            self._successors[u].append(v)
            self._predecessors[v].append(u)
        # --- END OPTIMIZATION ---

        # Iterate through vertices in reverse perfect elimination order


        processed_vertices = set()

        for v in reversed(self._order):
        # for v in self._order:

            # --- OPTIMIZATION: Use pre-computed lists ---
            # These lookups are much faster than calling graph.predecessors()
            predecessors_of_v = self._predecessors[v]
            successors_of_v = self._successors[v]

            # remove processed vertices from predecessors and successors
            predecessors_of_v = [u for u in predecessors_of_v if u not in processed_vertices]
            successors_of_v = [w for w in successors_of_v if w not in processed_vertices]

            # Consider all paths of the form u -> v -> w
            for u in predecessors_of_v:
                for w in successors_of_v:
                    # if u or w is already processed, skip this iteration
                    if u in processed_vertices or w in processed_vertices:
                        # print("skipped")
                        continue
                    

                    _complexity_counter += 1

                    # --- OPTIMIZATION: Direct dictionary access ---
                    weight_uv = self._weights[u][v]
                    weight_vw = self._weights[v][w]
                    # --- END OPTIMIZATION ---
                    
                    path_weight = weight_uv + weight_vw

                    if u == w:
                        if path_weight < 0:
                            self._pre_singular_vertices.add(u)
                    else:
                        # --- OPTIMIZATION: Direct dictionary access ---
                        # defaultdict makes this cleaner and faster than get_edge_data
                        current_weight_uw = self._weights[u][w]
                        # --- END OPTIMIZATION ---

                        if path_weight < current_weight_uw:
                            # --- OPTIMIZATION: Update fast structures directly ---
                            is_new_edge = self._weights[u][w] == PLUS_INF_WEIGHT
                            self._weights[u][w] = path_weight
                            
                            if is_new_edge:
                                # This is a new fill-in edge, update adjacency lists
                                _new_edges_counter += 1
                                self._successors[u].append(w)
                                self._predecessors[w].append(u)

                                if (u, v) not in self._all_edges_set: 
                                    pass
                                    raise ValueError(f"Edge ({u}, {v}) not found in the original graph edges set.")
                            # --- END OPTIMIZATION ---
                            
                            self._fill_in_edge_to_parent_edge_pair[(u, w)] = ((u, v), (v, w))
            processed_vertices.add(v)
        logging.debug("Complexity counter: %d", _complexity_counter)
        logging.debug("New edges counter: %d", _new_edges_counter)


    def _compute_singular_vertices(self):
        """
        This method computes the singular vertices based on the filled graph.
        It now builds a temporary networkx graph from the optimized internal
        data structures to run the SCC algorithm.
        """
        self._singular_vertices.clear()

        if not self._pre_singular_vertices:
            return

        # --- OPTIMIZATION: Build a temporary graph for SCC computation ---
        # This is a necessary step as the SCC algorithm is complex and best
        # left to the networkx library. We build the graph from our fast
        # adjacency list.
        graph_filled_for_scc = nx.DiGraph()
        graph_filled_for_scc.add_nodes_from(self._successors.keys())
        for u, successors in self._successors.items():
            for v in successors:
                graph_filled_for_scc.add_edge(u, v)
        # --- END OPTIMIZATION ---

        sccs = nx.strongly_connected_components(graph_filled_for_scc)

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
        Returns a vector of shortest path distances from token0 to all other tokens.
        This is an optimized version that operates on pre-computed dictionaries
        and lists for maximum speed, avoiding networkx overhead in loops.
        """
        # --- OPTIMIZATION: Check for node existence in our own structure ---
        if token0 not in self._successors:
            raise ValueError(f"Token {token0} not found in the graph.")
        # --- END OPTIMIZATION ---

        # Initialize the distance and parent dictionaries.
        D = {node: PLUS_INF_WEIGHT for node in self._successors}
        D[token0] = 0.0
        
        parents = {}
        if need_parents:
            parents = {node: None for node in self._successors}
        
        vertex_to_index = {vertex: i for i, vertex in enumerate(self._order)}
        
        s_index = vertex_to_index.get(token0)
        # Handle case where token0 might be in the graph but not in the elimination order
        # (e.g., a disconnected component not covered by the treewidth decomposition)
        if s_index is None:
             # Fallback for disconnected nodes: only paths to self are known
            for node in D:
                if D[node] > PLUS_INF_WEIGHT / 2 and node != token0:
                    D[node] = PLUS_INF_WEIGHT
            return D, parents


        # --- First Pass: Downward Sweep ---
        for k_idx in range(s_index, -1, -1):
            k = self._order[k_idx]
            
            # --- OPTIMIZATION: Use fast, pre-computed successors and weights ---
            for j in self._successors[k]:
                if j in vertex_to_index:
                    j_idx = vertex_to_index[j]
                    if j_idx < k_idx:
                        new_dist = D[k] + self._weights[k][j]
                        if new_dist < D[j]:
                            D[j] = new_dist
                            if need_parents:
                                parents[j] = k
            # --- END OPTIMIZATION ---

        # --- Second Pass: Upward Sweep ---
        for k_idx in range(len(self._order)):
            k = self._order[k_idx]
            
            # --- OPTIMIZATION: Use fast, pre-computed successors and weights ---
            for j in self._successors[k]:
                if j in vertex_to_index:
                    j_idx = vertex_to_index[j]
                    if j_idx > k_idx:
                        new_dist = D[k] + self._weights[k][j]
                        if new_dist < D[j]:
                            D[j] = new_dist
                            if need_parents:
                                parents[j] = k
            # --- END OPTIMIZATION ---
                            
        for node in D:
            if D[node] > PLUS_INF_WEIGHT / 2:
                D[node] = PLUS_INF_WEIGHT
        return D, parents


    def query_path(self, token0, token1):
        """
        Queries the shortest path from token0 to token1 by first finding the
        path in the filled graph and then recursively expanding the fill-in edges.
        This adapted method is robust and relies on the previously optimized components.
        """
        # Step 1: Call SSSP to get distances and parent pointers.
        # This is the main performance dependency. We handle the case where the
        # source token might not be in the graph.
        try:
            D, parents = self.query_sssp(token0, need_parents=True)
        except ValueError:
            # This occurs if token0 is not in the graph.
            return {
                'distances': PLUS_INF_WEIGHT,
                'path': [],
            }

        # Step 2: Check if a path to the destination exists in the filled graph.
        distance = D.get(token1, PLUS_INF_WEIGHT)
        if distance >= PLUS_INF_WEIGHT:
            # This also covers the case where token1 is not in the graph.
            return { 
                'distances': PLUS_INF_WEIGHT,
                'path': [],
            }

        # Handle the trivial case of a path to itself.
        if token0 == token1:
            return {
                'distances': 0.0,
                'path': [token0]
            }

        # Step 3: Reconstruct the path in the filled graph using the parent pointers.
        # This path may contain "shortcut" (fill-in) edges that need to be expanded.
        filled_path_nodes = []
        curr = token1
        while curr is not None:
            filled_path_nodes.append(curr)
            curr = parents.get(curr)
        filled_path_nodes.reverse()

        # Basic validation: the reconstructed path should start with token0.
        if not filled_path_nodes or filled_path_nodes[0] != token0:
            return {
                'distances': PLUS_INF_WEIGHT,
                'path': [],
            }

        # Step 4: Iteratively expand the shortcut edges to find the true path.
        # We use a stack to manage the path segments (edges) to be processed.
        final_path = [token0]
        
        # Create a stack of edges from the filled path.
        work_stack = []
        for i in range(len(filled_path_nodes) - 1):
            work_stack.append((filled_path_nodes[i], filled_path_nodes[i+1]))
        
        # Reverse the stack so that when we pop(), we get the first segment of the path,
        # allowing us to build the final_path in the correct order.
        work_stack.reverse()

        while work_stack:
            u, v = work_stack.pop()

            # Check if the edge (u, v) is a fill-in/shortcut edge. This is a fast dictionary lookup.
            if (u, v) in self._fill_in_edge_to_parent_edge_pair:
                # If it's a shortcut, it was formed from two parent edges.
                # We expand it by pushing its parent edges back onto the stack.
                (edge1, edge2) = self._fill_in_edge_to_parent_edge_pair[(u, v)]
                
                # The second edge is pushed first so that the first edge (from u) is at the
                # top of the stack and will be processed next, maintaining path order.
                work_stack.append(edge2)
                work_stack.append(edge1)
            else:
                # If it's an original edge, its destination is the next node in our final path.
                # The logic relies on the fact that the start of this edge `u` is the
                # end of the previously expanded primitive edge.
                final_path.append(v)
                
        return {
            'distances': distance,
            'path': final_path
        }

# this is for testing 
# it generates random graph with no negative cycles
def random_weighted_no_negative_cycles(n: int, p: float, positive_magnitude: float, jonsons_magnitude: float, random_seed: int) -> nx.DiGraph:
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
    G = nx.fast_gnp_random_graph(n, p, directed=True, seed=random_seed)

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
    n = 50  # Number of nodes
    p = 0.04  # Probability of edge creation
    positive_magnitude = 1.0  # Maximum positive weight
    jonsons_magnitude = 5.0  # Magnitude for potential vector

    

    # let's create the APSP matrix 

    up = UniPrice()

    for trial in range(1000):

        G = random_weighted_no_negative_cycles(n, p, positive_magnitude, jonsons_magnitude, random_seed=random.randint(0, 1000000000000))
        
        if need_draw:
            # print all edges in the graph 
            for u, v, data in G.edges(data=True):
                print(f"Edge ({u}, {v}) with weight {data['weight']}")

            # draw without weights
            # import matplotlib.pyplot as plt
            # pos = nx.spring_layout(G)
            # nx.draw(G, pos, with_labels=True, node_size=700, node_color='lightblue', font_size=10, font_color='black', edge_color='gray')
            # # NO WEIGHTS
            # plt.title("Random Directed Graph with Positive Weights")
            # plt.show()

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
                    print(f"Discrepancy found for {key}: {dist1} vs {dist2}")
                    # write_gpickle(G, f"debug_graph_{trial}.gpickle")

                    # save the graph to a file for debugging
                    raise ValueError(f"Discrepancy found for {key}: {dist1} vs {dist2}")
    

        except nx.NetworkXUnbounded:
            logging.error("negative cycle detected, cannot compute all-pairs shortest paths.")

        print(f"Test {trial + 1} passed.")

        # let's find all-pairs shortest paths 

