import json
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import random
import itertools
from lib.graph_analysis import GraphAnalyzer

class WalkFriendlyGenerator:
    def __init__(self, features, n=6):
        self.features = features
        self.n = n
        
    def generate(self):
        """
        Generates a graph optimized for Walk < TSP using JSON features.
        Creates a structure where Walk can use cheap paths but TSP must use expensive ones.
        """
        # Extract features
        mean_w = self.features.get('mean_weight', 50)
        min_w = self.features.get('min_weight', 10)
        max_w = self.features.get('max_weight', 90)
        low_cost_density = self.features.get('low_cost_density', 0.3)
        
        # Initialize matrix with high costs
        matrix = np.full((self.n, self.n), float(mean_w))
        np.fill_diagonal(matrix, 0)
        
        # Strategy: Create a "path-friendly" structure
        # 1. Build a random spanning tree with low costs (Walk can follow this)
        # 2. Make non-tree edges expensive (TSP forced to use these for cycle)
        
        # Generate random spanning tree
        nodes = list(range(self.n))
        tree_edges = []
        remaining = nodes[1:]
        connected = [nodes[0]]
        
        while remaining:
            # Connect a remaining node to any connected node
            new_node = random.choice(remaining)
            connect_to = random.choice(connected)
            tree_edges.append(tuple(sorted((new_node, connect_to))))
            connected.append(new_node)
            remaining.remove(new_node)
        
        # Get all possible edges
        all_edges = list(itertools.combinations(range(self.n), 2))
        non_tree_edges = [e for e in all_edges if e not in tree_edges]
        
        # Set tree edges to low cost (Walk advantage) - keep them consistently low
        for i, j in tree_edges:
            low_cost = min_w + random.uniform(0, 1)  # Reduced variance
            low_cost = max(low_cost, 1)
            matrix[i][j] = low_cost
            matrix[j][i] = low_cost
            
        # Add a few more low-cost edges based on density
        num_extra_low = max(0, int(len(all_edges) * low_cost_density) - len(tree_edges))
        random.shuffle(non_tree_edges)
        
        for k in range(min(num_extra_low, len(non_tree_edges))):
            i, j = non_tree_edges[k]
            low_cost = min_w + random.uniform(0, 1)  # Reduced variance
            low_cost = max(low_cost, 1)
            matrix[i][j] = low_cost
            matrix[j][i] = low_cost
            
        # Set remaining edges to high cost (TSP disadvantage) - ensure bigger gap
        used_low_edges = tree_edges + non_tree_edges[:num_extra_low]
        for i, j in all_edges:
            if (i, j) not in used_low_edges and (j, i) not in used_low_edges:
                high_cost = mean_w + random.uniform(15, max_w - mean_w)  # Increased minimum
                high_cost = max(high_cost, min_w + 20)  # Ensure significant gap
                matrix[i][j] = high_cost
                matrix[j][i] = high_cost
        
        # Final adjustments
        matrix = np.clip(matrix, 1, 200)
        matrix = (matrix + matrix.T) / 2
        np.fill_diagonal(matrix, 0)
        
        return matrix

def main():
    if len(sys.argv) > 1:
        try:
            n = int(sys.argv[1])
        except ValueError:
            print("Usage: python walk_friendly_generator.py [N]")
            return
    else:
        n = 6

    print(f"Generating Walk-friendly graphs for N={n}...")

    # Load features
    try:
        json_path = os.path.join(os.path.dirname(__file__), '../data/walk_features.json')
        with open(json_path, 'r') as f:
            data = json.load(f)
            features = data.get('walk_better_features', {})
            if not features:
                print("Error: 'walk_better_features' not found in JSON.")
                return
    except FileNotFoundError:
        print("Error: 'walk_features.json' not found.")
        return

    print("Using features:")
    for k, v in features.items():
        print(f"  {k}: {v:.3f}")

    generator = WalkFriendlyGenerator(features, n=n)
    analyzer = GraphAnalyzer(n=n)

    # Generate and test a few graphs
    print(f"\nGenerating and testing graphs:")
    print("="*70)
    
    successes = 0
    total_trials = 1
    
    for i in range(total_trials):
        matrix = generator.generate()
        
        # Solve TSP and Walk
        tsp_cost, walk_cost = analyzer.solve(matrix)
        
        # Print first 3 matrices for inspection
        if i < 3:
            print(f"\nGraph {i+1} (N={n}) distance matrix:")
            print(np.array2string(matrix, precision=2, suppress_small=True, max_line_width=120))
            print(f"TSP cost   : {tsp_cost:.2f}")
            print(f"Walk cost  : {walk_cost:.2f}")
            ratio = tsp_cost / walk_cost if walk_cost > 0 else float('inf')
            print(f"Ratio (TSP/Walk): {ratio:.2f}x")
            
            if walk_cost < tsp_cost - 1e-9:
                print("✓ Walk is better!")
                successes += 1
            else:
                print("✗ TSP wins")
                
            print("-" * 60)
        else:
            # Count silently
            if walk_cost < tsp_cost - 1e-9:
                successes += 1
    
    success_rate = (successes / total_trials) * 100
    print(f"\nResults for N={n}:")
    print(f"Total Graphs: {total_trials}")
    print(f"Walk < TSP: {successes}")
    print(f"Success Rate: {success_rate:.1f}%")

if __name__ == "__main__":
    main()