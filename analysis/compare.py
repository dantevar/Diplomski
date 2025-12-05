import itertools
import networkx as nx
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from generators.NonMetricGraphGenerator import NonMetricGraphGenerator

class GraphModel:
    def __init__(self, distance_matrix):
        self.distance_matrix = distance_matrix
        self.n = len(distance_matrix)
        self.G = nx.DiGraph()
        for i in range(self.n):
            self.G.add_node(i)
        for i in range(self.n):
            for j in range(self.n):
                if i != j:
                    self.G.add_edge(i, j, weight=float(distance_matrix[i][j]))

    def metric_closure(self):
        """Vrati metric closure grafa"""
        mc = nx.DiGraph()
        for i in self.G.nodes():
            for j in self.G.nodes():
                if i != j:
                    cost = nx.shortest_path_length(self.G, i, j, weight="weight")
                    mc.add_edge(i, j, weight=cost)
        return mc

    def brute_force_tsp(self):
        """Standard TSP (Hamiltonian Cycle) using direct edges"""
        nodes = list(range(self.n))
        best_cost = float("inf")
        best_perm = None
        for perm in itertools.permutations(nodes):
            cost = 0.0
            for i in range(self.n):
                a = perm[i]
                b = perm[(i+1) % self.n]
                cost += self.distance_matrix[a][b]
            if cost < best_cost:
                best_cost = cost
                best_perm = perm
        return best_perm, best_cost

    def shortest_path_walk(self, perm):
        """Calculates cost of a walk following the given permutation using shortest paths"""
        walk_edges = []
        total_cost = 0.0
        for i in range(len(perm)):
            a = perm[i]
            b = perm[(i+1) % len(perm)]
            path = nx.shortest_path(self.G, a, b, weight="weight")
            cost = nx.shortest_path_length(self.G, a, b, weight="weight")
            # zapis bridova
            for j in range(len(path)-1):
                walk_edges.append( (path[j], path[j+1], self.distance_matrix[path[j]][path[j+1]]) )
            total_cost += cost
        return walk_edges, total_cost

    def brute_force_optimal_walk(self):
        """Finds the optimal walk (Generalized TSP) by solving TSP on Metric Closure"""
        mc = self.metric_closure()
        # Solve TSP on MC
        nodes = list(range(self.n))
        best_cost = float("inf")
        best_perm = None
        
        for perm in itertools.permutations(nodes):
            cost = 0.0
            for i in range(self.n):
                a = perm[i]
                b = perm[(i+1) % self.n]
                cost += mc[a][b]["weight"]
            if cost < best_cost:
                best_cost = cost
                best_perm = perm
        
        # Reconstruct walk
        walk_edges, _ = self.shortest_path_walk(best_perm)
        return best_perm, walk_edges, best_cost

def main():
    n = 8
    seed_base = 0
    max_trials = 1000
    
    print(f"Searching for a graph with N={n} satisfying:")
    print("1. Cost(TSP) != Cost(Optimal Walk)")
    print("2. Cost(Optimal Walk) != Cost(Walk from TSP Perm)")
    print("3. Cost(TSP) != Cost(Walk from TSP Perm)")
    print("4. Cost(TSP) <= 2 * Cost(Optimal Walk)")
    print("-" * 50)

    generator = NonMetricGraphGenerator(n=n, num_clusters=3, low_cost_range=(10, 30), high_cost_range=(40, 80))

    for t in range(max_trials):
        seed = seed_base + t
        generator.seed = seed
        mat = generator.generate()
        
        model = GraphModel(mat)
        
        # 1. Standard TSP (Hamiltonian Cycle)
        perm_tsp, cost_tsp = model.brute_force_tsp()
        
        # 2. Optimal Walk (Generalized TSP)
        perm_opt_walk, walk_opt, cost_opt_walk = model.brute_force_optimal_walk()
        
        # 3. Walk from TSP Permutation
        walk_tsp, cost_walk_from_tsp = model.shortest_path_walk(perm_tsp)
        
        # Check conditions
        # Distinct costs
        distinct = (abs(cost_tsp - cost_opt_walk) > 1e-9 and 
                    abs(cost_opt_walk - cost_walk_from_tsp) > 1e-9 and 
                    abs(cost_tsp - cost_walk_from_tsp) > 1e-9)
        
        # Ratio condition
        ratio_ok = cost_tsp <= 2 * cost_opt_walk
        
        if distinct and ratio_ok:
            print(f"\n[SUCCESS] Found satisfying graph at trial {t} (seed={seed})")
            print("Distance Matrix:")
            print(mat)
            print("-" * 30)
            print(f"1. Standard TSP Cost (Cycle): {cost_tsp:.4f}")
            print(f"   Permutation: {perm_tsp}")
            print("-" * 30)
            print(f"2. Optimal Walk Cost (Generalized TSP): {cost_opt_walk:.4f}")
            print(f"   Permutation (on Metric Closure): {perm_opt_walk}")
            print(f"   Full Walk: {[u for u, v, w in walk_opt] + [walk_opt[-1][1]]}")
            print("-" * 30)
            print(f"3. Walk from TSP Permutation Cost: {cost_walk_from_tsp:.4f}")
            print(f"   (Using shortest paths between nodes in TSP order)")
            print("-" * 30)
            print(f"Ratio TSP/OptimalWalk: {cost_tsp/cost_opt_walk:.4f}")
            break
    else:
        print("Could not find a satisfying graph in max trials.")

if __name__ == "__main__":
    main()
