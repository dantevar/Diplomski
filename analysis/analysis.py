import numpy as np
import time
import networkx as nx
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from lib.algorithms import GeneralizedTSPSolver
from generators.EuclideanGraphGenerator import EuclideanGraphGenerator
from generators.MetricFriendlyGraphGenerator import MetricFriendlyGraphGenerator
from generators.NonMetricGraphGenerator import NonMetricGraphGenerator
import main as mzwv # Importing the existing solver file

def create_star_graph(n):
    """
    Creates a graph with a central hub (node 0) connected to all other nodes with low cost.
    Edges between other nodes (leaves) have high cost.
    This forces the optimal path to visit the hub multiple times.
    """
    dist_matrix = np.zeros((n, n))
    hub = 0
    low_cost = 1
    high_cost = 100
    
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if i == hub or j == hub:
                dist_matrix[i][j] = low_cost
            else:
                dist_matrix[i][j] = high_cost
    return dist_matrix

def run_analysis():
    print("=== Generalized TSP Analysis ===\n")
    print("Complexity Analysis:")
    print("1. Nearest Neighbor: O(N^2)")
    print("2. Christofides: O(N^3) (dominated by Matching)")
    print("3. 2-Opt: O(k * N^2)")
    print("4. Metric Closure (APSP): O(N^3)\n")

    # --- Scenario 1: Non-Euclidean Graph (Metric Friendly) ---
    # print("--- Scenario 1: Non-Euclidean Graph (Triangle Inequality Violated) ---")
    # n = 10
    # print(f"Generating graph with {n} nodes...")
    
    # gen = MetricFriendlyGraphGenerator(n=n, num_clusters=3, intra_high=5, inter_low=100)
    # dist_matrix_non_euc = gen.generate()
    # dist_matrix_non_euc = (dist_matrix_non_euc + dist_matrix_non_euc.T) / 2
    # np.fill_diagonal(dist_matrix_non_euc, 0)
    
    # # Solve using Exact Method
    # print("\nRunning Exact Brute Force (Reference)...")
    # model = mzwv.Model("NonEuc", "1.0", distance_matrix=dist_matrix_non_euc)
    # start_time = time.time()
    # bf_perm, bf_walk, bf_cost = model.brute_force(start_node=0)
    # print(f"Exact Cost: {bf_cost:.4f}")
    # print(f"Time: {time.time() - start_time:.4f}s")
    
    # solver = GeneralizedTSPSolver(dist_matrix_non_euc)
    
    # print("\nRunning Christofides (on Metric Closure)...")
    # start_time = time.time()
    # chris_walk, chris_cost = solver.christofides()
    # print(f"Christofides Cost: {chris_cost:.4f}")
    # print(f"Time: {time.time() - start_time:.4f}s")
    # print(f"Accuracy vs Exact: {chris_cost/bf_cost:.4f}")

    # # --- Scenario 2: Star Graph (Forces Repeated Visits) ---
    # print("\n\n--- Scenario 2: Star Graph (Hub Structure) ---")
    # n_star = 6
    # print(f"Generating Star Graph with {n_star} nodes (Node 0 is Hub)...")
    # dist_matrix_star = create_star_graph(n_star)
    
    # print("\nRunning Exact Brute Force...")
    # model_star = mzwv.Model("Star", "1.0", distance_matrix=dist_matrix_star)
    # bf_perm_star, bf_walk_star, bf_cost_star = model_star.brute_force(start_node=0)
    # print(f"Exact Cost: {bf_cost_star:.4f}")
    # print(f"Exact Walk: {bf_walk_star}")
    
    # solver_star = GeneralizedTSPSolver(dist_matrix_star)
    
    # print("\nRunning Christofides (on Metric Closure)...")
    # chris_walk_star, chris_cost_star = solver_star.christofides()
    # print(f"Christofides Cost: {chris_cost_star:.4f}")
    # print(f"Christofides Walk: {chris_walk_star}")
    
    # # Check for repeated visits
    # if len(bf_walk_star) > n_star + 1:
    #     print("Result: Optimal solution visits the Hub multiple times (Generalized TSP behavior confirmed).")
    # else:
    #     print("Result: Hamiltonian cycle found (Unexpected for Star Graph).")

    # # --- Scenario 3: Euclidean Graph (Triangle Inequality Holds) ---
    # print("\n\n--- Scenario 3: Euclidean Graph (Triangle Inequality Holds) ---")
    # n_euc = 15
    # print(f"Generating Euclidean graph with {n_euc} nodes...")
    
    # euc_gen = EuclideanGraphGenerator(n=n_euc, seed=42)
    # dist_matrix_euc, points = euc_gen.generate()
    
    # solver_euc = GeneralizedTSPSolver(dist_matrix_euc)
    
    # print("\nRunning Christofides (on Metric Closure)...")
    # start_time = time.time()
    # chris_walk_euc, chris_cost_euc = solver_euc.christofides()
    # print(f"Christofides Cost: {chris_cost_euc:.4f}")
    # print(f"Time: {time.time() - start_time:.4f}s")
    
    # # Verify Hamiltonian Property
    # from collections import Counter
    # node_counts = Counter(chris_walk_euc[:-1])
    # max_visits = max(node_counts.values())
    
    # print(f"\nWalk Length: {len(chris_walk_euc)}")
    # print(f"Unique Nodes Visited: {len(node_counts)}")
    
    # if max_visits == 1 and len(node_counts) == n_euc:
    #     print("Verification Successful: Solution is a Hamiltonian Cycle.")
    # else:
    #     print(f"Verification Failed: Max visits {max_visits}")

    # --- Scenario 4: Guaranteed Non-Metric (Walk < TSP) ---
    print("\n\n--- Scenario 4: Guaranteed Non-Metric Graph (Chain of Clusters) ---")
    n_nm = 8 # Reduced from 12 to avoid long brute force time
    print(f"Generating strictly non-metric graph with {n_nm} nodes (Chain of Clusters)...")
    
    # Using 3 clusters to form a chain A-B-C. 
    # TSP must jump C->A (expensive). Walk goes A->B->C->B->A (cheap).
    nm_gen = NonMetricGraphGenerator(n=n_nm, num_clusters=3, low_cost_range=(1, 5), high_cost_range=(50, 100), seed=42)
    dist_matrix_nm = nm_gen.generate()
    
    # 1. Calculate Standard TSP Cost (Hamiltonian Cycle)
    print("Calculating Standard TSP Cost (Hamiltonian Cycle)...")
    model_nm = mzwv.Model("NonMetric", "1.0", distance_matrix=dist_matrix_nm)
    tsp_perm, tsp_cost = model_nm.brute_force_tsp()
    print(f"Standard TSP Cost: {tsp_cost:.4f}")
    
    # 2. Calculate Generalized TSP Cost (Walk)
    print("Calculating Generalized TSP Cost (Walk)...")
    solver_nm = GeneralizedTSPSolver(dist_matrix_nm)
    chris_walk_nm, chris_cost_nm = solver_nm.christofides()
    print(f"Generalized TSP (Christofides) Cost: {chris_cost_nm:.4f}")
    
    # 3. Compare
    diff = tsp_cost - chris_cost_nm
    print(f"Difference (TSP - Walk): {diff:.4f}")
    
    if chris_cost_nm < tsp_cost:
        print("SUCCESS: Generalized Walk is strictly cheaper than Standard TSP.")
        print(f"Improvement: {(diff / tsp_cost) * 100:.2f}%")
    else:
        print("FAILURE: Walk cost is not smaller than TSP cost.")

if __name__ == "__main__":
    run_analysis()
