import numpy as np
import networkx as nx
import itertools
import json
import random
import time
from algorithms import GeneralizedTSPSolver
import main as mzwv

class GraphAnalyzer:
    def __init__(self, n=6):
        self.n = n

    def generate_random_graph(self):
        """Generates a random symmetric distance matrix."""
        # Using a wide range to allow for triangle inequality violations
        matrix = np.random.randint(1, 100, size=(self.n, self.n))
        matrix = (matrix + matrix.T) // 2
        np.fill_diagonal(matrix, 0)
        return matrix

    def extract_features(self, matrix):
        """Extracts structural features from the graph."""
        n = len(matrix)
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                edges.append(matrix[i][j])
        edges = np.array(edges)
        
        # 1. Triangle Inequality Violations
        violations = 0
        total_triplets = 0
        violation_magnitudes = []
        
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    if i != j and j != k and i != k:
                        total_triplets += 1
                        if matrix[i][k] > matrix[i][j] + matrix[j][k]:
                            violations += 1
                            violation_magnitudes.append(matrix[i][k] - (matrix[i][j] + matrix[j][k]))
        
        violation_ratio = violations / total_triplets if total_triplets > 0 else 0
        violation_severity = np.mean(violation_magnitudes) if violation_magnitudes else 0
        
        # 2. Edge Statistics
        mean_weight = float(np.mean(edges))
        std_weight = float(np.std(edges))
        cv = std_weight / mean_weight if mean_weight > 0 else 0
        min_weight = float(np.min(edges))
        max_weight = float(np.max(edges))
        
        # MST Features
        G = nx.Graph()
        for i in range(n):
            for j in range(i + 1, n):
                G.add_edge(i, j, weight=matrix[i][j])
        mst = nx.minimum_spanning_tree(G)
        mst_cost = mst.size(weight='weight')
        mst_ratio = mst_cost / mean_weight if mean_weight > 0 else 0

        # 3. Advanced Statistics
        # Skewness
        if std_weight > 0:
            skewness = float(np.mean(((edges - mean_weight) / std_weight) ** 3))
            kurtosis = float(np.mean(((edges - mean_weight) / std_weight) ** 4)) - 3
        else:
            skewness = 0.0
            kurtosis = 0.0
            
        # Low cost density (edges significantly below mean)
        low_cost_threshold = mean_weight - 0.5 * std_weight
        low_cost_count = sum(1 for e in edges if e < low_cost_threshold)
        low_cost_density = low_cost_count / len(edges) if len(edges) > 0 else 0
        
        return {
            "violation_ratio": violation_ratio,
            "violation_severity": violation_severity,
            "mean_weight": mean_weight,
            "std_weight": std_weight,
            "cv": cv,
            "min_weight": min_weight,
            "max_weight": max_weight,
            "mst_cost": mst_cost,
            "mst_ratio": mst_ratio,
            "skewness": skewness,
            "kurtosis": kurtosis,
            "low_cost_density": low_cost_density
        }

    def solve(self, matrix):
        """Solves TSP and Generalized TSP (Walk)."""
        # 1. Standard TSP (Brute Force)
        model = mzwv.Model("Temp", "1.0", distance_matrix=matrix)
        # Suppress output if possible, or just ignore
        tsp_perm, tsp_cost = model.brute_force_tsp()
        
        # 2. Generalized TSP (Walk) - Using Christofides on Metric Closure for speed/approx
        # OR Brute Force on Metric Closure for exactness (since N=6, BF is fast)
        solver = GeneralizedTSPSolver(matrix)
        # Let's implement a quick BF on Metric Closure here to be exact
        mc = solver.metric_closure_matrix
        nodes = list(range(self.n))
        best_walk_cost = float('inf')
        
        for perm in itertools.permutations(nodes):
            cost = 0.0
            for i in range(self.n):
                a = perm[i]
                b = perm[(i+1) % self.n]
                cost += mc[a][b]
            if cost < best_walk_cost:
                best_walk_cost = cost
                
        return tsp_cost, best_walk_cost

class FeatureBasedGenerator:
    def __init__(self, features, n=6):
        self.features = features
        self.n = n
        
    def generate(self):
        """
        Generates a graph attempting to match the target features.
        Uses a structural approach to ensure triangle inequality violations.
        """
        # Extract features from the provided dictionary
        mean_w = self.features.get('mean_weight', 50)
        base_min_w = self.features.get('min_weight', 10)
        max_w = self.features.get('max_weight', 90)
        low_cost_density = self.features.get('low_cost_density', 0.3)
        
        # Add variance to min_weight instead of using fixed value
        min_w = base_min_w + random.uniform(-3, 3)
        
        # Start with a base matrix with high costs (max_w)
        # Using float to avoid integer overflow/type issues
        matrix = np.full((self.n, self.n), float(max_w))
        np.fill_diagonal(matrix, 0)
        
        # Add random noise to base
        noise = np.random.randint(-5, 5, size=(self.n, self.n))
        matrix = matrix + noise
        
        # To achieve violation ratio, we need "shortcuts".
        # Use low_cost_density to determine how many edges should be 'min_w'
        num_edges = self.n * (self.n - 1) // 2
        num_low = int(num_edges * low_cost_density)
        
        # Ensure at least a few shortcuts if density is too low but we want violations
        if num_low < self.n and self.features.get('violation_ratio', 0) > 0.05:
             num_low = self.n

        # Jednostavan pristup: nasumično rasporedom jeftinih bridova
        all_pairs = list(itertools.combinations(range(self.n), 2))
        random.shuffle(all_pairs)
        
        # Primijeni jeftine bridove na prvih num_low parova s varijacijom
        for k in range(num_low):
            i, j = all_pairs[k]
            # Add some noise to low-cost edges instead of fixed min_w
            low_val = min_w + random.uniform(-2, 2)
            low_val = max(low_val, 1)  # Keep positive
            matrix[i][j] = low_val
            matrix[j][i] = low_val
            
        # Izračunaj vrijednost za skupe bridove da pogodi prosjek
        if num_edges > num_low:
            target_high = (mean_w * num_edges - min_w * num_low) / (num_edges - num_low)
        else:
            target_high = max_w

        # Dodijeli skupe bridove s više varijabilnosti
        for k in range(num_low, num_edges):
            i, j = all_pairs[k]
            # Increase variance in high-cost edges
            val = target_high + random.uniform(-15, 15)
            val = max(val, min_w + 5)  # osiguraj da je visok > nizak
            matrix[i][j] = val
            matrix[j][i] = val

        # Final clip and symmetry
        matrix = np.clip(matrix, 1, 200) # Keep positive
        matrix = (matrix + matrix.T) / 2
        np.fill_diagonal(matrix, 0)
            
        return matrix

def run_analysis():
    sizes = list(range(4, 11))
    trials_per_size = 200
    
    all_walk_better_features = []
    all_tsp_equal_features = []
    
    print(f"Starting analysis for N={sizes}...")
    
    for n in sizes:
        print(f"\nGenerating {trials_per_size} random graphs (N={n})...")
        analyzer = GraphAnalyzer(n=n)
        
        for _ in range(trials_per_size):
            matrix = analyzer.generate_random_graph()
            tsp_cost, walk_cost = analyzer.solve(matrix)
            features = analyzer.extract_features(matrix)
            
            if walk_cost < tsp_cost - 1e-9:
                all_walk_better_features.append(features)
            else:
                all_tsp_equal_features.append(features)
                
    total_graphs = len(sizes) * trials_per_size
    print(f"\nAnalysis Results (Aggregated N={sizes}):")
    print(f"Total Graphs: {total_graphs}")
    print(f"Walk Better: {len(all_walk_better_features)} ({len(all_walk_better_features)/total_graphs*100:.1f}%)")
    
    if not all_walk_better_features:
        print("No graphs found where Walk is better.")
        return

    # Calculate average features
    def avg_feats(feats_list):
        if not feats_list: return {}
        keys = feats_list[0].keys()
        return {k: np.mean([f[k] for f in feats_list]) for k in keys}
    
    avg_walk = avg_feats(all_walk_better_features)
    avg_tsp = avg_feats(all_tsp_equal_features)
    
    print("\nFeature Comparison (Average):")
    print(f"{'Feature':<20} | {'Walk Better':<15} | {'TSP Equal':<15} | {'Diff':<10}")
    print("-" * 65)
    for k in avg_walk:
        v1 = avg_walk[k]
        v2 = avg_tsp.get(k, 0)
        print(f"{k:<20} | {v1:<15.4f} | {v2:<15.4f} | {v1-v2:<10.4f}")
        
    # Save features
    output_data = {
        "walk_better_features": avg_walk,
        "tsp_equal_features": avg_tsp
    }
    with open('walk_features.json', 'w') as f:
        json.dump(output_data, f, indent=4)
    print("\nSaved average features of both graph types to 'walk_features.json'.")
    
    # Generate new graphs based on features
    test_n = 8
    print(f"\nGenerating new graphs (N={test_n}) based on extracted features...")
    generator = FeatureBasedGenerator(avg_walk, n=test_n)
    
    new_walk_better_count = 0
    new_trials = 100
    analyzer_test = GraphAnalyzer(n=test_n)
    
    for _ in range(new_trials):
        matrix = generator.generate()
        tsp_cost, walk_cost = analyzer_test.solve(matrix)
        if walk_cost < tsp_cost - 1e-9:
            new_walk_better_count += 1
            
    print(f"New Generation Success Rate (N={test_n}): {new_walk_better_count}/{new_trials} ({new_walk_better_count/new_trials*100:.1f}%)")

if __name__ == "__main__":
    run_analysis()
