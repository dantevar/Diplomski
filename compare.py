#!/usr/bin/env python3
import itertools
from typing import List, Tuple
import networkx as nx
import random

def generate_random_distance_matrix(n: int, low: int = 1, high: int = 20, seed: int = None):
    """
    Generira simetričan distance matrix (int) sa značajnijim razlikama između
    direktnih edge-ova i mogućih shortcutova. Cilj: metric closure TSP put
    može biti jeftiniji od direktnog Hamiltonovog ciklusa.
    """
    if seed is not None:
        random.seed(seed)

    # 1. Random integer edges s većim rasponom
    mat = [[0 if i == j else random.randint(low, high) for j in range(n)] for i in range(n)]

    # 2. Simetriziraj
    for i in range(n):
        for j in range(i + 1, n):
            mat[j][i] = mat[i][j]

    # 3. Dodaj snažne "shortcutove" da biasira TSP vs shortest-path
    num_shortcuts = max(1, n // 2)
    for _ in range(num_shortcuts):
        i, j = random.sample(range(n), 2)
        # biramo treći čvor k da spojimo jeftinim shortcutom
        k = random.choice([x for x in range(n) if x != i and x != j])
        # shortcut značajno jeftiniji od direktnog i klasičnih edge-ova
        cheap_cost = max(low, mat[i][j] // 3, mat[i][k] // 2, mat[k][j] // 2)
        mat[i][k] = cheap_cost
        mat[k][i] = cheap_cost
        mat[k][j] = cheap_cost
        mat[j][k] = cheap_cost

    return mat


class GraphModel:
    def __init__(self, distance_matrix: List[List[int]]):
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
                    cost = nx.shortest_path_length(self.G, i, j, weight='weight')
                    mc.add_edge(i, j, weight=cost)
        return mc

    def brute_force_tsp(self):
        nodes = list(range(self.n))
        best_cost = float('inf')
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

    def shortest_path_walk(self, perm: Tuple[int, ...]):
        walk_edges = []
        total_cost = 0.0
        for i in range(len(perm)):
            a = perm[i]
            b = perm[(i+1) % len(perm)]
            path = nx.shortest_path(self.G, a, b, weight='weight')
            cost = nx.shortest_path_length(self.G, a, b, weight='weight')
            # zapis bridova
            for j in range(len(path)-1):
                walk_edges.append( (path[j], path[j+1], self.distance_matrix[path[j]][path[j+1]]) )
            total_cost += cost
        return walk_edges, total_cost

    def brute_force_shortestpath(self):
        nodes = list(range(self.n))
        best_cost = float('inf')
        best_perm = None
        best_walk = None
        for perm in itertools.permutations(nodes):
            walk_edges, cost = self.shortest_path_walk(perm)
            if cost < best_cost:
                best_cost = cost
                best_perm = perm
                best_walk = walk_edges
        return best_perm, best_walk, best_cost


if __name__ == "__main__":
    n = 4
    seed_base = 32000000
    trials = 200
    count = 0
    for t in range(trials):
        seed = seed_base + t
        mat = generate_random_distance_matrix(n=n, seed=seed)

        model = GraphModel(mat)
        
        # 1. Metric closure
        mc = model.metric_closure()
        mc_model = GraphModel(mat)
        mc_model.G = mc
        perm_mc, cost_mc = mc_model.brute_force_tsp()
        walk_mc, walk_cost_mc = model.shortest_path_walk(perm_mc)

        # 2. Original graph “brute-force TSP-like”
        perm_orig, walk_orig, cost_orig = model.brute_force_shortestpath()

        # 3. Original graph TSP then shortest-path walk
        perm_tsp, cost_tsp = model.brute_force_tsp()
        walk_tsp, walk_cost_tsp = model.shortest_path_walk(perm_tsp)

        # provjera
        if not (abs(walk_cost_mc - cost_orig) < 1e-9 and abs(cost_orig - walk_cost_tsp) < 1e-9):
            count += 1
            print(f"\n[Trial seed={seed}] Distance matrix with cost mismatch:")
            for row in mat:
                print(row)
            print("\nMetric closure perm:", perm_mc)
            print("Walk in original graph:", walk_mc)
            print("Cost:", walk_cost_mc)

            print("\nOriginal graph brute-force-like perm:", perm_orig)
            print("Walk:", walk_orig)
            print("Cost:", cost_orig)

            print("\nOriginal graph TSP perm:", perm_tsp)
            print("Walk:", walk_tsp)
            print("Cost:", walk_cost_tsp)

    print(f"\nPercentage of mismatches over {trials} trials: {(count / trials) * 100:.2f}%")