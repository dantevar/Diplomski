import networkx as nx
import numpy as np
import itertools

class GeneralizedTSPSolver:
    def __init__(self, distance_matrix):
        self.original_matrix = np.array(distance_matrix)
        self.n = len(distance_matrix)
        self.G = nx.Graph() # Undirected for Christofides (usually defined for undirected metric TSP)
        # Note: If the original graph is directed/asymmetric, Christofides is not directly applicable 
        # without modification or assuming symmetry. 
        # The problem statement mentions "complete weighted graph" which usually implies undirected unless specified.
        # However, main.py used DiGraph. 
        # If the graph is directed, we can't easily use Christofides (which relies on MST and Matching).
        # But "Metric Closure" makes it symmetric if the original graph allows moving back and forth with same cost?
        # No, shortest path u->v might be different from v->u in directed graph.
        # But standard TSP usually assumes undirected or we use Asymmetric TSP solvers.
        # Given "Problem trgovačkog putnika" usually refers to the symmetric version in standard curriculum unless "Asymmetric" is specified.
        # Also "težina brida" (edge weight) usually implies undirected edges in Croatian math terminology ("brid" vs "luk").
        # I will assume the metric closure should be treated as symmetric for Christofides, 
        # or I will implement a version that works on the symmetric closure if the graph is undirected.
        # Let's assume the input distance matrix represents an undirected graph (symmetric) or we treat it as such for the base algorithms.
        # If the user provided asymmetric inputs in main.py, I should be careful.
        # Let's check main.py again. It uses `nx.DiGraph`.
        # However, the user prompt says "težina brida" (edge) not "luka" (arc).
        # I will implement Christofides for the Symmetric case.
        # For the general case, I'll compute the metric closure. If the metric closure is asymmetric, Christofides is not applicable.
        # I will assume symmetric metric closure for Christofides.
        
        self._build_graph()
        self.metric_closure_matrix = self._compute_metric_closure()

    def _build_graph(self):
        for i in range(self.n):
            for j in range(self.n):
                if i != j:
                    self.G.add_edge(i, j, weight=self.original_matrix[i][j])

    def _compute_metric_closure(self):
        # Compute All-Pairs Shortest Paths
        # Using Floyd-Warshall or Johnson's. NetworkX has floyd_warshall_numpy
        # But we need to reconstruct paths too.
        
        # Let's use nx.all_pairs_dijkstra for path reconstruction support
        # We need a matrix of distances and a way to reconstruct paths.
        
        # For the algorithms, we mainly need the distance matrix of the closure.
        # We will reconstruct the path at the end.
        
        G_dir = nx.DiGraph()
        for i in range(self.n):
            for j in range(self.n):
                if i != j:
                    G_dir.add_edge(i, j, weight=self.original_matrix[i][j])
        
        closure_matrix = np.zeros((self.n, self.n))
        self.shortest_paths = {}
        
        for i in range(self.n):
            lengths, paths = nx.single_source_dijkstra(G_dir, i, weight='weight')
            self.shortest_paths[i] = paths
            for j in range(self.n):
                if j in lengths:
                    closure_matrix[i][j] = lengths[j]
                else:
                    closure_matrix[i][j] = float('inf')
                    
        return closure_matrix

    def reconstruct_path(self, tour):
        """
        Convert a tour (sequence of nodes) in the metric closure 
        back to a walk in the original graph.
        """
        walk = []
        for k in range(len(tour) - 1):
            u, v = tour[k], tour[k+1]
            # Get shortest path from u to v
            if v in self.shortest_paths[u]:
                path_segment = self.shortest_paths[u][v]
                # Append path (excluding the start node to avoid duplication, 
                # except for the very first node of the whole walk)
                if k == 0:
                    walk.extend(path_segment)
                else:
                    walk.extend(path_segment[1:])
            else:
                raise ValueError(f"No path from {u} to {v}")
        return walk

    def calculate_walk_cost(self, walk):
        cost = 0
        for i in range(len(walk) - 1):
            u, v = walk[i], walk[i+1]
            cost += self.original_matrix[u][v]
        return cost

    # --- Algorithms ---

    def nearest_neighbor(self):
        """
        Nearest Neighbor Heuristic on the Metric Closure.
        """
        unvisited = set(range(self.n))
        start_node = 0
        current = start_node
        tour = [current]
        unvisited.remove(current)
        
        while unvisited:
            next_node = min(unvisited, key=lambda x: self.metric_closure_matrix[current][x])
            tour.append(next_node)
            unvisited.remove(next_node)
            current = next_node
            
        tour.append(start_node) # Return to start
        
        walk = self.reconstruct_path(tour)
        cost = self.calculate_walk_cost(walk)
        return walk, cost

    def christofides(self):
        """
        Christofides Algorithm (1.5-approximation) on the Metric Closure.
        Requires the metric closure to be symmetric (undirected graph).
        """
        # Check symmetry
        if not np.allclose(self.metric_closure_matrix, self.metric_closure_matrix.T):
            print("Warning: Metric closure is not symmetric. Christofides might not be applicable/optimal.")
            # We can proceed by treating it as undirected (using average or min), 
            # or just use the upper triangle.
            # For this implementation, we'll build an undirected graph from the closure.
        
        # 1. Create a complete graph from metric closure
        G_closure = nx.Graph()
        for i in range(self.n):
            for j in range(i + 1, self.n):
                G_closure.add_edge(i, j, weight=self.metric_closure_matrix[i][j])
                
        # 2. Minimum Spanning Tree
        T = nx.minimum_spanning_tree(G_closure, weight='weight')
        
        # 3. Find odd degree nodes in T
        odd_degree_nodes = [v for v, d in T.degree() if d % 2 == 1]
        
        # 4. Minimum Weight Perfect Matching on odd degree nodes
        # Create subgraph of G_closure induced by odd_degree_nodes
        subgraph = G_closure.subgraph(odd_degree_nodes)
        # Note: nx.min_weight_matching returns a set of edges
        # We need to negate weights because networkx calculates MAX weight matching
        # Or use max_weight_matching with negated weights.
        # Actually, nx has min_weight_matching since v2.2? No, it's usually max_weight_matching.
        # Let's check available functions.
        # Usually we invert weights: new_weight = max_weight - weight
        
        # Create a new graph for matching with inverted weights
        G_matching = nx.Graph()
        max_w = 0
        for u, v, d in subgraph.edges(data=True):
            if d['weight'] > max_w:
                max_w = d['weight']
                
        for u, v, d in subgraph.edges(data=True):
            G_matching.add_edge(u, v, weight= -(d['weight'])) # Negate for max weight matching to find min cost
            
        matching = nx.max_weight_matching(G_matching, maxcardinality=True)
        
        # 5. Add matching edges to T to form a multigraph
        M = nx.MultiGraph(T)
        for u, v in matching:
            weight = self.metric_closure_matrix[u][v]
            M.add_edge(u, v, weight=weight)
            
        # 6. Find Eulerian Circuit
        # M is guaranteed to be Eulerian (all degrees even)
        eulerian_circuit = list(nx.eulerian_circuit(M, source=0))
        
        # 7. Shortcut to Hamiltonian Circuit (Skip repeated vertices)
        tour = []
        visited = set()
        for u, v in eulerian_circuit:
            if u not in visited:
                tour.append(u)
                visited.add(u)
        tour.append(0) # Return to start
        
        walk = self.reconstruct_path(tour)
        cost = self.calculate_walk_cost(walk)
        return walk, cost

    def two_opt(self, initial_tour=None):
        """
        2-opt Local Search on the Metric Closure.
        """
        if initial_tour is None:
            # Start with Nearest Neighbor
            tour_nodes = self.nearest_neighbor()[0] 
            # Extract just the tour nodes (this is a walk, we need the tour on closure)
            # Actually, let's just re-run NN logic to get the tour indices
            unvisited = set(range(self.n))
            current = 0
            tour = [0]
            unvisited.remove(0)
            while unvisited:
                next_node = min(unvisited, key=lambda x: self.metric_closure_matrix[current][x])
                tour.append(next_node)
                unvisited.remove(next_node)
                current = next_node
            tour.append(0)
        else:
            tour = initial_tour

        improved = True
        while improved:
            improved = False
            for i in range(1, len(tour) - 2):
                for j in range(i + 1, len(tour) - 1):
                    if j - i == 1: continue # No point swapping adjacent edges
                    
                    # Old edges: (i-1, i) and (j, j+1)
                    # New edges: (i-1, j) and (i, j+1)
                    
                    old_cost = self.metric_closure_matrix[tour[i-1]][tour[i]] + \
                               self.metric_closure_matrix[tour[j]][tour[j+1]]
                    new_cost = self.metric_closure_matrix[tour[i-1]][tour[j]] + \
                               self.metric_closure_matrix[tour[i]][tour[j+1]]
                               
                    if new_cost < old_cost:
                        # Perform 2-opt swap
                        tour[i:j+1] = tour[i:j+1][::-1]
                        improved = True
        
        walk = self.reconstruct_path(tour)
        cost = self.calculate_walk_cost(walk)
        return walk, cost

